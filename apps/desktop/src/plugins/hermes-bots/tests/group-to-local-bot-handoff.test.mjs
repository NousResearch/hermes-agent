import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'

// #89834 — local open invokes registered group main-tab closer before canonical open.
const pluginSource = readFileSync(new URL('../plugin.js', import.meta.url), 'utf8')
const atom = i => {
  let v = i
  return { get: () => v, set: n => { v = typeof n === 'function' ? n(v) : n }, listen: () => () => {} }
}
const jsx = (t, p = {}) => ({ type: t, props: p })
const find = (tree, pred) => {
  if (tree == null || typeof tree !== 'object') return null
  if (Array.isArray(tree)) {
    for (const c of tree) { const f = find(c, pred); if (f) return f }
    return null
  }
  return pred(tree) ? tree : find(tree.props?.children, pred)
}

function load() {
  const timeline = [], notifications = []
  const host = {
    state: {
      profile: { get: () => 'other', listen: () => {} },
      gateway: { get: () => 'open', listen: () => {} },
      connectionId: { get: () => 'local', listen: () => {} }
    },
    request: async m => m === 'session.create'
      ? { stored_session_id: 'stored-chat', session_id: 'rt-chat' }
      : { profiles: [], sessions: [] },
    notify: p => notifications.push(p),
    notifyError: () => {}, openSession: async () => {}, ensureAgent: async () => {},
    activeConnectionId: () => 'local', warmAgent: () => {}, warmProfile: () => {},
    newChat: () => timeline.push({ type: 'newChat' }),
    navigate: () => timeline.push({ type: 'navigate' })
  }
  const ui = 'Button Checkbox Codicon ConfirmDialog ContextMenu ContextMenuContent ContextMenuItem ContextMenuSeparator ContextMenuTrigger CopyButton Dialog DialogContent DialogDescription DialogFooter DialogHeader DialogTitle DropdownMenu DropdownMenuContent DropdownMenuItem DropdownMenuTrigger EmptyState GlyphSpinner Input ScrollArea SearchField Select SelectContent SelectItem SelectTrigger SelectValue Switch Textarea Tip'.split(' ')
  const ctx = {
    atom, jsx, jsxs: jsx, cn: (...a) => a.filter(Boolean).join(' '), haptic: () => {},
    useEffect: () => {}, useRef: i => ({ current: typeof i === 'function' ? i() : i }),
    useState: i => [typeof i === 'function' ? i() : i, () => {}],
    useValue: s => (s && typeof s.get === 'function' ? s.get() : s),
    useQuery: () => ({ data: [], isLoading: false, isFetching: false, refetch: () => {} }),
    ...Object.fromEntries(ui.map(n => [n, n])),
    PALETTE_AREA: 'palette', COMPOSER_AREAS: { middleware: 'middleware' },
    profileColor: () => '#000', queryClient: { invalidateQueries: () => {} }, relativeTime: () => 'now',
    document: { getElementById: () => null, createElement: () => ({}), head: { appendChild: () => {} } },
    setTimeout, clearTimeout, console, Date, Math, JSON, Promise, Map, Set, URL, Error,
    Array, Object, String, Boolean, Number, RegExp, host
  }
  const code = pluginSource
    .replace(/^import\s+\*\s+as\s+sdk\s+from '@hermes\/plugin-sdk'\r?\n/m, '')
    .replace(/^import\s+\{[\s\S]*?\}\s+from '@hermes\/plugin-sdk'\r?\n/m, '')
    .replace(/^const \{ McpTab, ToolsetConfigPanel \} = sdk\r?\n/m, '')
    .replace(/^import .* from 'react'\r?\n/m, '')
    .replace(/^import .* from 'react\/jsx-runtime'\r?\n/m, '')
    .replace('export default {', 'globalThis.plugin = {')
    .concat(`\nglobalThis.__h={BotRow,ActiveNowStrip,BotsPane,openGroupChat,groupChatMainTabs,$groupChatWorkspace};`)
  vm.runInNewContext(code, ctx, { filename: 'plugin.js' })
  // Live binding: handlers resolve this mutable global at call time.
  ctx.openBotCanonicalChat = async (...a) => { timeline.push({ type: 'openBotCanonicalChat', args: a }); return 'stored-chat' }
  const api = ctx.__h
  const registerGroup = group => {
    let closed = 0
    ctx.host.openWorkspace = () => () => { closed += 1; timeline.push({ type: 'mainTabClose', group }) }
    api.openGroupChat(group)
    assert.equal(api.$groupChatWorkspace.get(), group)
    assert.ok(api.groupChatMainTabs.has(group))
    return { closed: () => closed, registered: () => api.groupChatMainTabs.has(group) }
  }
  const botRow = bot => {
    const tree = api.BotRow({ bot, onEdit: () => {} })
    const row = tree.type === 'button' ? tree : tree.props.children[0].props.children
    assert.equal(typeof row.props.onClick, 'function')
    return row
  }
  // Real Active Now onOpen from BotsPane (not a hand-copied fragment).
  const strip = find(api.BotsPane(), n => n?.props && typeof n.props.onOpen === 'function' && 'roster' in n.props)
  assert.ok(strip, 'BotsPane mounts ActiveNowStrip with onOpen')
  const onOpen = strip.props.onOpen
  const activeChip = bot => {
    const b = { ...bot, last_session: bot.last_session || { id: 'live', last_active: Math.floor(Date.now() / 1000) - 5 } }
    const tree = api.ActiveNowStrip({ roster: [b], activeProfile: 'other', gatewayState: 'open', metaByName: {}, onOpen })
    const chip = find(tree, n => n.type === 'button' && /Open /.test(String(n.props?.title || ''))
      && (String(n.props.title).includes(bot.name) || String(n.props.title).includes(bot.title || '')))
    assert.ok(chip, 'Active Now chip renders')
    return chip
  }
  return { api, timeline, notifications, registerGroup, botRow, activeChip }
}

const drain = async () => { await new Promise(r => setImmediate(r)); await new Promise(r => setImmediate(r)) }
const assertCloseBeforeOpen = tl => {
  const c = tl.findIndex(e => e.type === 'mainTabClose'), o = tl.findIndex(e => e.type === 'openBotCanonicalChat')
  assert.ok(c >= 0, 'registered main-tab closer must run')
  assert.ok(o >= 0, 'local open must reach canonical chat')
  assert.ok(c < o, 'close must precede canonical open')
}
const remote = { name: 'research', title: 'Research', remoteSource: true, sourceScoped: true, connectionId: 'work', connectionLabel: 'Work' }

test('BotRow local open closes group main tab before canonical open', async () => {
  const r = load(), tab = r.registerGroup('Core')
  await r.botRow({ name: 'alpha', title: 'Alpha' }).props.onClick()
  assert.equal(tab.closed(), 1, 'registered main-tab closer must run')
  assert.equal(r.api.$groupChatWorkspace.get(), null)
  assert.equal(tab.registered(), false)
  assertCloseBeforeOpen(r.timeline)
})

test('BotRow remote open does not dismiss group main tab', async () => {
  const r = load(), tab = r.registerGroup('Core')
  await r.botRow(remote).props.onClick()
  assert.equal(tab.closed(), 0, 'remote open must not dismiss the group tab')
  assert.equal(r.api.$groupChatWorkspace.get(), 'Core')
  assert.equal(tab.registered(), true)
  assert.equal(r.timeline.some(e => e.type === 'openBotCanonicalChat'), false)
  assert.ok(r.notifications.some(n => /Stay in this chat/.test(n.message || '')))
})

test('Active Now local open closes group main tab before canonical open', async () => {
  const r = load(), tab = r.registerGroup('Ops')
  await r.activeChip({ name: 'alpha', title: 'Alpha' }).props.onClick()
  await drain()
  assert.equal(tab.closed(), 1, 'registered main-tab closer must run')
  assert.equal(r.api.$groupChatWorkspace.get(), null)
  assert.equal(tab.registered(), false)
  assertCloseBeforeOpen(r.timeline)
})

test('Active Now remote open does not dismiss group main tab', async () => {
  const r = load(), tab = r.registerGroup('Ops')
  await r.activeChip(remote).props.onClick(); await drain()
  assert.equal(tab.closed(), 0, 'remote open must not dismiss the group tab')
  assert.equal(r.api.$groupChatWorkspace.get(), 'Ops')
  assert.equal(tab.registered(), true)
})

test('no group / old-host: local open is safe and still opens chat', async () => {
  const r = load()
  assert.equal(r.api.$groupChatWorkspace.get(), null)
  await assert.doesNotReject(() => r.botRow({ name: 'alpha', title: 'Alpha' }).props.onClick())
  assert.ok(r.timeline.some(e => e.type === 'openBotCanonicalChat'))
  r.api.$groupChatWorkspace.set('Legacy')
  assert.equal(r.api.groupChatMainTabs.has('Legacy'), false)
  await r.botRow({ name: 'beta', title: 'Beta' }).props.onClick()
  assert.equal(r.api.$groupChatWorkspace.get(), null)
})
