import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'

const pluginSource = readFileSync(new URL('../plugin.js', import.meta.url), 'utf8')
const atom = initial => {
  let value = initial
  return { get: () => value, set: next => { value = typeof next === 'function' ? next(value) : next }, listen: () => () => {} }
}
const jsx = (type, props = {}) => ({ type, props })
const find = (tree, predicate) => {
  if (tree == null || typeof tree !== 'object') return null
  if (Array.isArray(tree)) {
    for (const child of tree) { const match = find(child, predicate); if (match) return match }
    return null
  }
  return predicate(tree) ? tree : find(tree.props?.children, predicate)
}

function load({ rejectOpen = false, rejectPrepare = false } = {}) {
  const timeline = []
  const host = {
    state: {
      profile: { get: () => 'default', listen: () => {} },
      gateway: { get: () => 'open', listen: () => {} },
      connectionId: { get: () => 'local', listen: () => {} }
    },
    request: async () => ({ profiles: [], sessions: [] }),
    notify: () => {}, notifyError: error => timeline.push({ type: 'error', error }),
    openSession: async () => {}, ensureAgent: async () => {}, requestProfile: async () => ({}),
    activeConnectionId: () => 'local', warmAgent: () => {}, warmProfile: () => {},
    newChat: () => timeline.push({ type: 'newChat' }), navigate: () => {}
  }
  const ui = 'Button Checkbox Codicon ConfirmDialog ContextMenu ContextMenuContent ContextMenuItem ContextMenuSeparator ContextMenuTrigger CopyButton Dialog DialogContent DialogDescription DialogFooter DialogHeader DialogTitle DropdownMenu DropdownMenuContent DropdownMenuItem DropdownMenuTrigger EmptyState GlyphSpinner Input ScrollArea SearchField Select SelectContent SelectItem SelectTrigger SelectValue Switch Textarea Tip'.split(' ')
  const context = {
    atom, jsx, jsxs: jsx, cn: (...values) => values.filter(Boolean).join(' '), haptic: () => {},
    useEffect: () => {}, useMemo: fn => fn(), useRef: value => ({ current: typeof value === 'function' ? value() : value }),
    useState: value => [typeof value === 'function' ? value() : value, () => {}],
    useValue: store => store?.get ? store.get() : store,
    useQuery: () => ({ data: [], isLoading: false, isFetching: false, refetch: () => {} }),
    ...Object.fromEntries(ui.map(name => [name, name])),
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
    .concat('\nglobalThis.__h={BotRow,BotsPane,openRosterBot,openGroupChat,groupChatMainTabs,$groupChatWorkspace,$groupChats};')
  vm.runInNewContext(code, context, { filename: 'plugin.js' })
  context.openBotCanonicalChat = async bot => {
    timeline.push({ type: 'canonicalOpen', bot })
    if (rejectOpen) throw new Error('canonical open failed')
    return { registryId: 'stored-chat', openedId: 'stored-chat' }
  }
  if (rejectPrepare) context.prepareBotSource = async () => { throw new Error('source preparation failed') }
  const api = context.__h
  const registerGroup = group => {
    api.$groupChats.set({ ...api.$groupChats.get(), [group]: { log: [], watermarks: {}, sessions: {}, stranded: {} } })
    host.openWorkspace = id => () => timeline.push({ type: 'workspaceClose', id })
    api.openGroupChat(group)
    return { closed: () => timeline.filter(event => event.type === 'workspaceClose' && event.id.includes(':group:')).length }
  }
  const botRow = bot => {
    const tree = api.BotRow({ bot, onEdit: () => {} })
    return tree.type === 'button' ? tree : tree.props.children[0].props.children
  }
  const activeOnOpen = find(api.BotsPane(), node => node?.props && typeof node.props.onOpen === 'function' && 'roster' in node.props).props.onOpen
  return { api, timeline, registerGroup, botRow, activeOnOpen }
}

const drain = async () => { await new Promise(resolve => setImmediate(resolve)); await new Promise(resolve => setImmediate(resolve)) }
const bot = { name: 'alpha', title: 'Alpha' }
const assertCloseBeforeOpen = runtime => {
  const closed = runtime.timeline.findIndex(event => event.type === 'workspaceClose' && event.id.includes(':group:'))
  const opened = runtime.timeline.findIndex(event => event.type === 'canonicalOpen')
  assert.ok(closed >= 0 && opened >= 0 && closed < opened)
}

test('central owner open closes the registered group tab before canonical open', async () => {
  const runtime = load(), tab = runtime.registerGroup('Core')
  assert.equal(await runtime.api.openRosterBot(bot), true)
  assert.equal(tab.closed(), 1)
  assert.equal(runtime.api.groupChatMainTabs.has('Core'), false)
  assertCloseBeforeOpen(runtime)
})

test('BotRow and Active Now both delegate through the central owner open', async () => {
  for (const invoke of [runtime => runtime.botRow(bot).props.onClick(), runtime => runtime.activeOnOpen(bot)]) {
    const runtime = load(); runtime.registerGroup('Core'); invoke(runtime); await drain(); assertCloseBeforeOpen(runtime)
  }
})

test('failed canonical open restores the group fallback and surfaces an error', async () => {
  const runtime = load({ rejectOpen: true }), tab = runtime.registerGroup('Core')
  assert.equal(await runtime.api.openRosterBot(bot), false)
  assert.equal(tab.closed(), 1)
  assert.equal(runtime.api.$groupChatWorkspace.get(), 'Core')
  assert.equal(runtime.api.groupChatMainTabs.has('Core'), true)
  assert.ok(runtime.timeline.some(event => event.type === 'error'))
})

test('failed source preparation restores the group fallback', async () => {
  const runtime = load({ rejectPrepare: true }); runtime.registerGroup('Core')
  assert.equal(await runtime.api.openRosterBot({ ...bot, sourceScoped: true, connectionId: 'local' }), false)
  assert.equal(runtime.api.$groupChatWorkspace.get(), 'Core')
})

test('owner open with no selected group is safe', async () => {
  const runtime = load()
  assert.equal(await runtime.api.openRosterBot(bot), true)
  assert.equal(runtime.timeline.some(event => event.type === 'workspaceClose' && event.id.includes(':group:')), false)
})
