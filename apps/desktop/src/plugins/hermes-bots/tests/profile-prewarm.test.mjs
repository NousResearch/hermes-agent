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

function renderBotRow(name = 'alpha') {
  const botRowSource = sourceBetween('function BotRow(', '// ── model picker')
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
    cn: (...values) => values.filter(Boolean).join(' '),
    createCanonicalChat: async () => null,
    displayName: bot => bot.name,
    duplicateBot: async () => `${name}-copy`,
    haptic: () => undefined,
    // #49 session-aware-row helpers referenced inside BotRow.
    previewKind: () => ({ fromBot: false, sender: null }),
    generatedSessionTitle: () => null,
    ACTIVE_WINDOW_S: 90,
    A2A_PREFIX_RE: /^$/,
    useEffect: () => undefined,
    useState: initial => [typeof initial === 'function' ? initial() : initial, () => undefined],
    host: {
      state: { gateway: atom('open'), profile: atom('default') },
      warmProfile: profile => warmed.push(profile),
      request: async () => ({ sessions: [] }),
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

  vm.runInNewContext(`${botRowSource}\nglobalThis.BotRow = BotRow`, context)

  const tree = context.BotRow({ bot: { name }, onEdit: context.onEdit })
  const row = tree.props.children[0].props.children

  return { row, warmed }
}

test('regression: rendering BotsPane does not prewarm the entire roster', () => {
  const botsPaneSource = sourceBetween('function BotsPane(', '// ── plugin')

  assert.doesNotMatch(botsPaneSource, /host\.warmProfile/)
})

test('behavior: pointer entry prewarms only the hovered bot', () => {
  const { row, warmed } = renderBotRow('alpha')

  assert.deepEqual(warmed, [])
  assert.equal(typeof row.props.onPointerEnter, 'function')
  row.props.onPointerEnter()
  assert.deepEqual(warmed, ['alpha'])
})
