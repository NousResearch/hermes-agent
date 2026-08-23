import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'

const source = readFileSync(new URL('../plugin.js', import.meta.url), 'utf8')

function sourceBetween(start, end) {
  const from = source.indexOf(start)
  const to = source.indexOf(end, from)

  assert.notEqual(from, -1, `missing ${start}`)
  assert.notEqual(to, -1, `missing ${end}`)

  return source.slice(from, to)
}

// BotRow's activity resolver — extract the REAL helper so the harness can't
// drift from production behavior.
function activitySessionSource() {
  return sourceBetween('function botActivitySession(', '/** Bots that are working')
}

function renderBotRow(input = 'alpha') {
  const bot = typeof input === 'string' ? { name: input } : input
  const name = bot.name
  const botRowSource = sourceBetween('function BotRow(', '// ── model picker')
  // REAL owner-route resolver — the derivation from bare connectionId rows is
  // exactly what these tests exercise, so a hand stub would drift.
  const routeSource = sourceBetween('function botConnectionRoute(', 'function rewriteCliProfileOperands(')
  const ensured = []
  const opened = []
  const saved = []
  const warmed = []
  let liveConnectionId = 'local'
  const atom = value => ({
    get: () => value,
    set: next => {
      value = next
    }
  })
  const node = (type, props = {}) => ({ type, props })
  const context = {
    BotFace: 'BotFace',
    ContextMenu: 'ContextMenu',
    ContextMenuContent: 'ContextMenuContent',
    ContextMenuItem: 'ContextMenuItem',
    ContextMenuSeparator: 'ContextMenuSeparator',
    ContextMenuTrigger: 'ContextMenuTrigger',
    Codicon: 'Codicon',
    Tip: 'Tip',
    ROSTER_KEY: ['hermes-bots', 'roster'],
    $botMeta: atom({}),
    $botUnread: atom({}),
    $botAttention: atom({}),
    BOT_ATTENTION_HINTS: {},
    $botChatFocused: atom(false),
    $botsHomeFronted: atom(false),
    $focusedBotOwner: atom({ connectionId: 'local', profile: 'default' }),
    $focusedBotProfile: atom('default'),
    $groupChatWorkspace: atom(null),
    $lastRoster: atom([]),
    $selectedBot: atom('default'),
    $selectedRosterKey: atom(''),
    botAppearance: () => ({ shape: 'round', color: '#000', image: null }),
    botMetaKey: value => value.sourceScoped
      ? `${value.route?.connectionId || value.connectionId}::${value.name}`
      : value.name,
    botGroups: () => [],
    botHandle: value => value,
    botRosterKey: value => `${value.connectionId || 'legacy'}::${value.name}`,
    botRowOwnsWorkspace: () => false,
    botSourceStatus: () => ({ available: true, label: 'Ready' }),
    botOpenGeneration: 0,
    botRosterMeta: (_bot, metaByName) => {
      const key = _bot.sourceScoped
        ? `${_bot.route?.connectionId || _bot.connectionId}::${_bot.name}`
        : _bot.name
      return metaByName?.[key] ?? null
    },
    cn: (...values) => values.filter(Boolean).join(' '),
    createCanonicalChat: async () => null,
    displayName: bot => bot.name,
    duplicateBot: async () => `${name}-copy`,
    ensureBotMetadata: async () => ({ pinned: true }),
    haptic: () => undefined,
    // #49 session-aware-row helpers referenced inside BotRow.
    previewKind: () => ({ fromBot: false, sender: null }),
    generatedSessionTitle: () => null,
    focusedRosterOwner: owner => ({ connectionId: owner.connectionId, name: owner.profile }),
    isActiveRosterBot: () => false,
    isBackfilledFacePng: () => false,
    isBotHidden: () => false,
    isBotPinned: () => false,
    botSelectionKey: value => value.sourceScoped ? `${value.connectionId}::${value.name}` : value.name,
    isDefaultBot: value => value.name === 'default',
    newBotChat: () => undefined,
    openBotCanonicalChat: async (...args) => {
      opened.push(args)
      return 'stored-chat'
    },
    openRosterBot: async value => {
      opened.push([value])
      return true
    },
    workerActiveAt: () => false,
    ACTIVE_WINDOW_S: 90,
    A2A_PREFIX_RE: /^$/,
    useEffect: () => undefined,
    useState: initial => [typeof initial === 'function' ? initial() : initial, () => undefined],
    host: {
      state: { gateway: atom('open'), profile: atom('default') },
      ensureAgent: async (connectionId, profile) => {
        ensured.push([connectionId, profile])
        liveConnectionId = connectionId
      },
      activeConnectionId: () => liveConnectionId,
      requestProfile: async (_route, method) =>
        method === 'profiles.list'
          ? { profiles: [{ name, ui_meta: { 'hermes-bots': { chat: 'owner-chat' } } }] }
          : { sessions: [] },
      warmAgent: (connectionId, profile) => warmed.push([connectionId, profile]),
      warmProfile: profile => warmed.push(profile),
      request: async method =>
        method === 'profiles.list'
          ? { profiles: [{ name, ui_meta: { 'hermes-bots': { chat: 'owner-chat' } } }] }
          : { sessions: [] },
      notify: () => undefined,
      notifyError: () => undefined
    },
    jsx: node,
    jsxs: node,
    onEdit: () => undefined,
    persistBotMetaSnapshot: () => Promise.resolve(),
    queryClient: { invalidateQueries: () => undefined },
    relativeTime: () => 'now',
    requestForBot: async (_bot, method) => method === 'profiles.list'
      ? {
          profiles: [{
            name: _bot.route?.targetProfile || _bot.name,
            ui_meta: { 'hermes-bots': { pinned: true } }
          }]
        }
      : {},
    saveBotMeta: (_bot, patch) => saved.push([_bot, patch]),
    showsHandle: () => false,
    stripPreviewMarkdown: text => String(text || ''),
    useValue: store => store.get()
  }

  vm.runInNewContext(`${activitySessionSource()}\n${routeSource}\n${botRowSource}\nglobalThis.BotRow = BotRow`, context)

  const tree = context.BotRow({ bot, onEdit: context.onEdit })
  const row = tree.type === 'button' ? tree : tree.props.children[0].props.children

  return { ensured, opened, row, saved, tree, warmed }
}

test('regression: rendering BotsPane does not prewarm the entire roster', () => {
  const botsPaneSource = sourceBetween('function BotsPane(', '// ── plugin')

  assert.doesNotMatch(botsPaneSource, /host\.warmProfile/)
})

test('regression: source transitions keep BotRow hook order stable', () => {
  const botRowSource = sourceBetween('function BotRow(', '// ── model picker')

  assert.match(botRowSource, /const unreadByName = useValue\(\$botUnread\)/)
  assert.doesNotMatch(botRowSource, /remoteSource && Boolean\(useValue/)
})

test('behavior: pointer entry prewarms only the hovered bot', () => {
  const { row, warmed } = renderBotRow('alpha')

  assert.deepEqual(warmed, [])
  assert.equal(typeof row.props.onPointerEnter, 'function')
  row.props.onPointerEnter()
  assert.deepEqual(warmed, ['alpha'])
})

test('behavior: context-menu pin mutation hydrates a non-identity alias before toggling', async () => {
  const bot = {
    connectionId: 'remote-a',
    name: 'worker',
    remoteSource: true,
    sourceScoped: true,
    route: {
      connectionId: 'remote-a',
      mode: 'remote',
      profile: 'worker',
      targetProfile: 'backend-worker'
    }
  }
  const { saved, tree } = renderBotRow(bot)
  const menuItems = tree.props.children[1].props.children

  menuItems[0].props.onSelect()
  await new Promise(resolve => setTimeout(resolve, 0))

  assert.equal(saved.length, 1)
  assert.equal(saved[0][0], bot)
  assert.deepEqual(JSON.parse(JSON.stringify(saved[0][1])), { pinned: false })
})

test('behavior: a remote Connections row opens through its captured route without activation authority', async () => {
  const { ensured, opened, row, warmed } = renderBotRow({
    connectionId: 'work',
    connectionLabel: 'Work',
    name: 'research',
    remoteSource: true,
    sourceScoped: true
  })

  row.props.onPointerEnter()
  assert.deepEqual(warmed, [['work', 'research']])

  await row.props.onClick()
  assert.deepEqual(ensured, [])
  assert.equal(opened.length, 1)
  assert.equal(opened[0][0].connectionId, 'work')
})

test('behavior: remote default never opens the same-name local chat', async () => {
  const bot = {
    connectionId: 'mac-mini',
    connectionLabel: 'Mac Mini',
    name: 'default',
    remoteSource: true,
    sourceScoped: true
  }
  const botRowSource = sourceBetween('function BotRow(', '// ── model picker')
  const ensured = []
  const opened = []
  const errors = []
  const atom = value => ({
    get: () => value,
    set: next => {
      value = next
    }
  })
  const node = (type, props = {}) => ({ type, props })
  const context = {
    BotFace: 'BotFace',
    ContextMenu: 'ContextMenu',
    ContextMenuContent: 'ContextMenuContent',
    ContextMenuItem: 'ContextMenuItem',
    ContextMenuSeparator: 'ContextMenuSeparator',
    ContextMenuTrigger: 'ContextMenuTrigger',
    Codicon: 'Codicon',
    Tip: 'Tip',
    ROSTER_KEY: ['hermes-bots', 'roster'],
    $botMeta: atom({ default: { chat: 'this-device-chat' } }),
    $botUnread: atom({}),
    $botAttention: atom({}),
    BOT_ATTENTION_HINTS: {},
    $botChatFocused: atom(false),
    $botsHomeFronted: atom(false),
    $focusedBotOwner: atom({ connectionId: 'mac-mini', profile: 'default' }),
    $focusedBotProfile: atom('default'),
    $groupChatWorkspace: atom(null),
    $lastRoster: atom([]),
    $selectedBot: atom('default'),
    $selectedRosterKey: atom(''),
    botAppearance: () => ({ shape: 'round', color: '#000', image: null }),
    botGroups: () => [],
    botHandle: value => value,
    botRosterKey: value => `${value.connectionId || 'legacy'}::${value.name}`,
    botRowOwnsWorkspace: () => false,
    botSourceStatus: () => ({ available: true, label: 'Ready' }),
    botOpenGeneration: 0,
    botRosterMeta: () => null,
    cn: (...values) => values.filter(Boolean).join(' '),
    createCanonicalChat: async () => null,
    displayName: bot => bot.connectionLabel || bot.name,
    duplicateBot: async () => 'copy',
    ensureBotMetadata: async () => ({ pinned: true }),
    haptic: () => undefined,
    previewKind: () => ({ fromBot: false, sender: null }),
    generatedSessionTitle: () => null,
    focusedRosterOwner: owner => ({ connectionId: owner.connectionId, name: owner.profile }),
    isActiveRosterBot: () => false,
    isBackfilledFacePng: () => false,
    isBotHidden: () => false,
    isBotPinned: () => false,
    botSelectionKey: value => value.sourceScoped ? `${value.connectionId}::${value.name}` : value.name,
    isDefaultBot: value => value.name === 'default',
    newBotChat: () => undefined,
    openBotCanonicalChat: async (...args) => {
      opened.push(args)
      return 'this-device-chat'
    },
    openRosterBot: async value => {
      opened.push([value])
      return true
    },
    workerActiveAt: () => false,
    ACTIVE_WINDOW_S: 90,
    A2A_PREFIX_RE: /^$/,
    useEffect: () => undefined,
    useState: initial => [typeof initial === 'function' ? initial() : initial, () => undefined],
    host: {
      state: { gateway: atom('open'), profile: atom('default') },
      ensureAgent: async (connectionId, profile) => ensured.push([connectionId, profile]),
      requestProfile: async () => ({}),
      activeConnectionId: () => 'local',
      warmAgent: () => undefined,
      warmProfile: () => undefined,
      request: async () => ({ profiles: [{ name: 'default', ui_meta: { 'hermes-bots': { chat: 'this-device-chat' } } }] }),
      notify: () => undefined,
      notifyError: (_err, msg) => errors.push(msg)
    },
    jsx: node,
    jsxs: node,
    onEdit: () => undefined,
    queryClient: { invalidateQueries: () => undefined },
    relativeTime: () => 'now',
    saveBotMeta: () => undefined,
    showsHandle: () => false,
    stripPreviewMarkdown: text => String(text || ''),
    useValue: store => store.get()
  }

  const routeSource = sourceBetween('function botConnectionRoute(', 'function rewriteCliProfileOperands(')
  vm.runInNewContext(`${activitySessionSource()}\n${routeSource}\n${botRowSource}\nglobalThis.BotRow = BotRow`, context)
  const tree = context.BotRow({ bot, onEdit: context.onEdit })
  const row = tree.type === 'button' ? tree : tree.props.children[0].props.children

  await row.props.onClick()
  assert.deepEqual(ensured, [])
  assert.equal(opened.length, 1)
  assert.equal(opened[0][0].connectionId, 'mac-mini')
  assert.equal(errors.length, 0)
})
