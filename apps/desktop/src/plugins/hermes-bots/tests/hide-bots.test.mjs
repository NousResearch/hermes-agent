import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'

// Hide and Pin are Desktop roster preferences. They are source-qualified,
// never mutate gateway profile metadata, and never change the active chat.
// Hidden bots keep working and remain available to mentions and group chats.

const pluginSource = readFileSync(new URL('../plugin.js', import.meta.url), 'utf8')

function load({ toastsOn = false } = {}) {
  const values = new Map()
  const atom = initial => {
    const slot = { get: () => values.get(slot), set: value => values.set(slot, value) }
    values.set(slot, initial)
    return slot
  }
  const notifications = []
  const requests = []
  const context = {
    atom,
    PALETTE_AREA: 'palette',
    COMPOSER_AREAS: { middleware: 'middleware' },
    document: { getElementById: () => null, createElement: () => ({}), head: { appendChild: () => undefined } },
    host: {
      state: {
        profile: { get: () => 'default', listen: () => undefined },
        gateway: { get: () => 'open', listen: () => undefined }
      },
      request: (method, params) => {
        requests.push({ method, params })
        return Promise.resolve({})
      },
      notify: params => notifications.push(params)
    },
    sdk: new Proxy({}, { get: () => undefined })
  }
  const source = pluginSource
    .replace(/^import\s+\*\s+as\s+sdk\s+from '@hermes\/plugin-sdk'\r?\n/m, '')
    .replace(/^import\s+\{[\s\S]*?\}\s+from '@hermes\/plugin-sdk'\r?\n/m, '')
    .replace(/^const \{ McpTab, ToolsetConfigPanel \} = sdk\r?\n/m, '')
    .replace(/^import .* from 'react'\r?\n/m, '')
    .replace(/^import .* from 'react\/jsx-runtime'\r?\n/m, '')
    .replace('export default {', 'globalThis.plugin = {')
    .concat(`
globalThis.__hide = {
  botRosterKey,
  isBotHidden,
  isBotPinned,
  saveRosterPreference,
  trackInboundActivity,
  $activityToasts,
  $botMeta,
  $botRosterPrefs,
  $botUnread,
  $lastRoster,
  $selectedBot,
  setPluginCtx: value => { pluginCtx = value }
};
`)
  vm.runInNewContext(source, context, { filename: 'plugin.js' })
  if (toastsOn) {
    context.__hide.$activityToasts.set(true)
  }
  return { ...context.__hide, notifications, requests }
}

test('hide persists a source-qualified Desktop preference without mutating a gateway', () => {
  const t = load()
  const writes = []
  t.setPluginCtx({ storage: { set: (key, value) => writes.push({ key, value }) } })
  const bot = { connectionId: 'local', name: 'researcher' }

  t.saveRosterPreference(bot, 'hidden', true)

  assert.equal(t.isBotHidden(bot, {}), true)
  assert.equal(writes.at(-1).key, 'bot-roster-preferences-v1')
  assert.equal(writes.at(-1).value['local::researcher'].hidden, true)
  assert.equal(t.requests.length, 0, 'a display preference must not call profiles.configure')
})

test('same-named profiles on different gateways have independent Hide and Pin state', () => {
  const t = load()
  const local = { connectionId: 'local', name: 'default' }
  const remote = { connectionId: 'studio', name: 'default', remoteSource: true }

  t.saveRosterPreference(remote, 'hidden', true)
  t.saveRosterPreference(local, 'pinned', true)

  assert.equal(t.isBotHidden(local, {}), false)
  assert.equal(t.isBotHidden(remote, {}), true)
  assert.equal(t.isBotPinned(local, {}), true)
  assert.equal(t.isBotPinned(remote, {}), false)
  assert.equal(t.botRosterKey(local), 'local::default')
  assert.equal(t.botRosterKey(remote), 'studio::default')
})

test('unhide writes an explicit false and keeps the current selection', () => {
  const t = load()
  const bot = { connectionId: 'work', name: 'writer', remoteSource: true }
  t.$selectedBot.set('default')

  t.saveRosterPreference(bot, 'hidden', true)
  t.saveRosterPreference(bot, 'hidden', false)

  assert.equal(t.isBotHidden(bot, {}), false)
  assert.equal(t.$botRosterPrefs.get()['work::writer'].hidden, false)
  assert.equal(t.$selectedBot.get(), 'default', 'Hide is not a routing or focus action')
})

test('legacy profile metadata remains a compatibility fallback until locally overridden', () => {
  const t = load()
  const bot = { connectionId: 'local', name: 'legacy' }
  const metadata = { legacy: { hidden: true, pinned: true } }

  assert.equal(t.isBotHidden(bot, metadata), true)
  assert.equal(t.isBotPinned(bot, metadata), true)

  t.saveRosterPreference(bot, 'hidden', false)
  assert.equal(t.isBotHidden(bot, metadata), false)
  assert.equal(t.isBotPinned(bot, metadata), true)
})

test('hidden activity still accumulates unread but never emits an activity toast', () => {
  const t = load({ toastsOn: true })
  const bot = { connectionId: 'local', name: 'writer' }
  t.saveRosterPreference(bot, 'hidden', true)
  t.$selectedBot.set('default')

  const rosterAt = ts => [
    { ...bot, last_session: { last_active: ts, preview: 'Message from editor: hi' } }
  ]
  t.trackInboundActivity(rosterAt(100))
  t.trackInboundActivity(rosterAt(200))

  assert.equal(t.$botUnread.get().writer, true)
  assert.equal(t.notifications.length, 0)
})

test('shape: visible and hidden display lists are derived without shrinking the live roster', () => {
  assert.match(pluginSource, /const hiddenBots = roster\.filter\(bot => isBotHidden\(bot, allMeta\)\)/)
  assert.match(pluginSource, /const visibleRoster = roster\.filter\(bot => !isBotHidden\(bot, allMeta\)\)/)
  assert.match(pluginSource, /const activeSourceRoster = roster\.filter\(bot => !bot\.remoteSource\)/)
  assert.match(pluginSource, /trackInboundActivity\(activeSourceRoster\)/)
})

test('shape: hidden bots recover from a folded section instead of a global reveal mode', () => {
  assert.match(pluginSource, /children: 'Hidden'/)
  assert.match(pluginSource, /'aria-expanded': hiddenExpanded/)
  assert.match(pluginSource, /onClick: \(\) => \$showHiddenBots\.set\(!hiddenExpanded\)/)
  assert.match(pluginSource, /children: hidden \? 'Unhide' : 'Hide'/)
  assert.doesNotMatch(pluginSource, /fallbackSelectionAfterHide/)
})

test('shape: mention resolution remains independent from roster visibility', () => {
  const mentions = pluginSource.slice(
    pluginSource.indexOf('function resolveRosterMentions('),
    pluginSource.indexOf('/** Source-qualified identity for a roster row')
  )
  assert.doesNotMatch(mentions, /hidden/i)
})
