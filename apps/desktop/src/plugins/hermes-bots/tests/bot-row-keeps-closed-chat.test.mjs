import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'

// A bot row click is "go to this bot", not "open its Bot Chat". Before this,
// every click resolved the canonical chat by name and opened it as a tab —
// with no record of a close anywhere (the plugin keeps no closed set; core's
// tile bucket only forgets), a Bot Chat the user closed came back beside every
// newer thread on every bot switch. Now a bot whose workspace already holds
// tabs comes back to the one the user left; the forever-chat is opened only
// when nothing is open, or on the explicit asks (row menu "Open Bot Chat",
// Bots home "Open chat").

const pluginSource = readFileSync(new URL('../plugin.js', import.meta.url), 'utf8')
const atom = initial => {
  let value = initial
  return { get: () => value, set: next => { value = typeof next === 'function' ? next(value) : next }, listen: () => () => {} }
}
const jsx = (type, props = {}) => ({ type, props })

function load({ focusHit = null, focusApi = true } = {}) {
  const timeline = []
  const host = {
    state: {
      profile: { get: () => 'default', listen: () => {} },
      gateway: { get: () => 'open', listen: () => {} },
      connectionId: { get: () => 'local', listen: () => {} }
    },
    request: async () => ({ profiles: [], sessions: [] }),
    notify: () => {}, notifyError: error => timeline.push({ type: 'error', error }),
    openSession: async (id, options) => timeline.push({ type: 'openSession', id, options }),
    ensureAgent: async () => {}, requestProfile: async () => ({}),
    activeConnectionId: () => 'local', warmAgent: () => {}, warmProfile: () => {},
    newChat: () => timeline.push({ type: 'newChat' }), navigate: () => {},
    setWorkspaceScope: (mode, ownerKey) => timeline.push({ type: 'scope', mode, ownerKey }),
    openWorkspace: () => () => timeline.push({ type: 'homeClose' })
  }
  if (focusApi) {
    host.focusOpenWorkspaceSession = ownerKey => {
      timeline.push({ type: 'focus', ownerKey })
      return focusHit
    }
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
    .concat('\nglobalThis.__h={openRosterBot,$openBotChat,$selectedRosterKey,$botsPaneVisible,openBotsHomeWorkspace};')
  vm.runInNewContext(code, context, { filename: 'plugin.js' })
  context.prepareBotSource = async bot => { timeline.push({ type: 'prepare', bot }) }
  context.openBotCanonicalChat = async bot => {
    timeline.push({ type: 'canonicalOpen', bot })
    return { registryId: 'bot-chat', openedId: 'bot-chat' }
  }
  return { api: context.__h, timeline, host }
}

const bot = { connectionId: 'local', name: 'alpha', title: 'Alpha' }
const types = runtime => runtime.timeline.map(event => event.type)

test('a row click comes back to the tab the bot already has open — no canonical open', async () => {
  const runtime = load({ focusHit: 'thread-2' })

  assert.equal(await runtime.api.openRosterBot(bot), true)
  assert.deepEqual(runtime.timeline.find(event => event.type === 'focus'), { type: 'focus', ownerKey: 'bot:alpha' })
  assert.ok(!types(runtime).includes('canonicalOpen'), 'the closed Bot Chat must not be resolved or re-opened')
  assert.ok(!types(runtime).includes('prepare'), 'open tabs need no source activation')
  assert.ok(!types(runtime).includes('openSession'))
  // The claim carries only the fronted tab: the focus edge it fires keeps the
  // claim (releaseStaleOpenBotChat), the home yields, and no registry id is
  // recorded because none was resolved.
  assert.deepEqual({ ...runtime.api.$openBotChat.get() }, { key: 'local::alpha', openedRegistryId: '', openedSessionId: 'thread-2' })
  assert.equal(runtime.api.$selectedRosterKey.get(), 'local::alpha')
})

test('a bot with nothing open still opens its canonical chat', async () => {
  const runtime = load({ focusHit: null })

  assert.equal(await runtime.api.openRosterBot(bot), true)
  assert.deepEqual(types(runtime).filter(type => type !== 'scope'), ['focus', 'prepare', 'canonicalOpen'])
  assert.equal(runtime.api.$openBotChat.get()?.openedRegistryId, 'bot-chat')
})

test('the explicit canonical ask skips the open-tab shortcut', async () => {
  const runtime = load({ focusHit: 'thread-2' })

  assert.equal(await runtime.api.openRosterBot(bot, { canonical: true }), true)
  assert.ok(!types(runtime).includes('focus'))
  assert.ok(types(runtime).includes('canonicalOpen'))
  assert.equal(runtime.api.$openBotChat.get()?.openedRegistryId, 'bot-chat')
})

test('an older shell without the focus API opens the canonical chat as before', async () => {
  const runtime = load({ focusApi: false, focusHit: 'thread-2' })

  assert.equal(await runtime.api.openRosterBot(bot), true)
  assert.deepEqual(types(runtime).filter(type => type !== 'scope'), ['prepare', 'canonicalOpen'])
})

test('a throwing focus API degrades to the canonical open', async () => {
  const runtime = load()
  runtime.host.focusOpenWorkspaceSession = () => { throw new Error('no tree yet') }

  assert.equal(await runtime.api.openRosterBot(bot), true)
  assert.ok(types(runtime).includes('canonicalOpen'))
})

test('the explicit asks pass canonical: true (row menu, Bots home)', () => {
  const menu = pluginSource.slice(pluginSource.indexOf("children: 'Open Bot Chat'") - 400, pluginSource.indexOf("children: 'Open Bot Chat'"))
  assert.match(menu, /openRosterBot\(bot, \{ canonical: true \}\)/)

  const home = pluginSource.slice(pluginSource.indexOf("children: 'Open chat'") - 400, pluginSource.indexOf("children: 'Open chat'"))
  assert.match(home, /openRosterBot\(bot, \{ canonical: true \}\)/)

  // A plain row click and the Active Now strip stay "go to this bot".
  assert.match(pluginSource, /const open = \(\) => void openRosterBot\(bot\)\n/)
  assert.match(pluginSource, /onOpen: bot => void openRosterBot\(bot\)\n/)
})

test('a reclaim of a fronted non-canonical tab does not re-resolve the Bot Chat', () => {
  const start = pluginSource.indexOf("host.onEvent('session.reclaimed'")
  const block = pluginSource.slice(start, pluginSource.indexOf('stopSidebarSync', start))
  const guard = block.indexOf('if (!claim.openedRegistryId)')
  assert.ok(guard > 0, 'reclaim listener guards on the registry id')
  assert.ok(guard < block.indexOf('openBotCanonicalChat(bot)'), 'the guard precedes the canonical re-open')
})
