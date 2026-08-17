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
    $lastRoster: atom([]),
    $selectedBot: atom('default'),
    botAppearance: () => ({ shape: 'round', color: '#000', image: null }),
    botHandle: value => value,
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
      ensureAgent: async (connectionId, profile) => ensured.push([connectionId, profile]),
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

test('behavior: a thin source row activates its owner and uses that owner\'s chat pin', async () => {
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
  assert.deepEqual(ensured, [['work', 'research']])
  assert.deepEqual(opened, [['research', 'owner-chat']])
})
