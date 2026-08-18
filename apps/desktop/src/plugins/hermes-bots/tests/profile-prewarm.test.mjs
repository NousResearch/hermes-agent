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

function renderBotRow(input = 'alpha') {
  const bot = typeof input === 'string' ? { name: input } : input
  const name = bot.name
  const prepareSource = sourceBetween('async function prepareBotSource(', 'function displayName(')
  const botRowSource = sourceBetween('function BotRow(', '// ── model picker')
  const ensured = []
  const opened = []
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
    ROSTER_KEY: ['hermes-bots', 'roster'],
    $botMeta: atom({}),
    $botUnread: atom({}),
    $focusedBotProfile: atom('default'),
    $groupChatWorkspace: atom(null),
    $lastRoster: atom([]),
    $selectedBot: atom('default'),
    botAppearance: () => ({ shape: 'round', color: '#000', image: null }),
    botGroups: () => [],
    botHandle: value => value,
    botOpenGeneration: 0,
    botRosterMeta: (_bot, metaByName) => metaByName?.[_bot.name] ?? null,
    cn: (...values) => values.filter(Boolean).join(' '),
    createCanonicalChat: async () => null,
    displayName: bot => bot.name,
    duplicateBot: async () => `${name}-copy`,
    haptic: () => undefined,
    // #49 session-aware-row helpers referenced inside BotRow.
    previewKind: () => ({ fromBot: false, sender: null }),
    generatedSessionTitle: () => null,
    openBotCanonicalChat: async (...args) => {
      opened.push(args)
      return 'stored-chat'
    },
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
    queryClient: { invalidateQueries: () => undefined },
    relativeTime: () => 'now',
    saveBotMeta: () => undefined,
    showsHandle: () => false,
    stripPreviewMarkdown: text => String(text || ''),
    useValue: store => store.get()
  }

  vm.runInNewContext(`${prepareSource}\n${botRowSource}\nglobalThis.BotRow = BotRow`, context)

  const tree = context.BotRow({ bot, onEdit: context.onEdit })
  const row = tree.type === 'button' ? tree : tree.props.children[0].props.children

  return { ensured, opened, row, warmed }
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

test('behavior: a remote Connections row stays in this chat instead of hopping SSH', async () => {
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
  assert.deepEqual(opened, [])
})

test('behavior: remote default does not open this-device chat when the source did not activate', async () => {
  const bot = {
    connectionId: 'mac-mini',
    connectionLabel: 'Mac Mini',
    name: 'default',
    remoteSource: true,
    sourceScoped: true
  }
  const prepareSource = sourceBetween('async function prepareBotSource(', 'function displayName(')
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
    ROSTER_KEY: ['hermes-bots', 'roster'],
    $botMeta: atom({ default: { chat: 'this-device-chat' } }),
    $botUnread: atom({}),
    $focusedBotProfile: atom('default'),
    $groupChatWorkspace: atom(null),
    $lastRoster: atom([]),
    $selectedBot: atom('default'),
    botAppearance: () => ({ shape: 'round', color: '#000', image: null }),
    botGroups: () => [],
    botHandle: value => value,
    botOpenGeneration: 0,
    botRosterMeta: () => null,
    cn: (...values) => values.filter(Boolean).join(' '),
    createCanonicalChat: async () => null,
    displayName: bot => bot.connectionLabel || bot.name,
    duplicateBot: async () => 'copy',
    haptic: () => undefined,
    previewKind: () => ({ fromBot: false, sender: null }),
    generatedSessionTitle: () => null,
    openBotCanonicalChat: async (...args) => {
      opened.push(args)
      return 'this-device-chat'
    },
    ACTIVE_WINDOW_S: 90,
    A2A_PREFIX_RE: /^$/,
    useEffect: () => undefined,
    useState: initial => [typeof initial === 'function' ? initial() : initial, () => undefined],
    host: {
      state: { gateway: atom('open'), profile: atom('default') },
      ensureAgent: async (connectionId, profile) => ensured.push([connectionId, profile]),
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

  vm.runInNewContext(`${prepareSource}\n${botRowSource}\nglobalThis.BotRow = BotRow`, context)
  const tree = context.BotRow({ bot, onEdit: context.onEdit })
  const row = tree.type === 'button' ? tree : tree.props.children[0].props.children

  await row.props.onClick()
  assert.deepEqual(ensured, [])
  assert.deepEqual(opened, [])
  assert.equal(errors.length, 0)
})
