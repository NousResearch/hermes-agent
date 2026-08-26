import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'

// Per-bot Hide/Unhide: right-click → Hide Bot persists `hidden: true` in bot
// meta (local ctx.storage + server ui_meta via saveBotMeta), hidden bots drop
// out of the roster list, and a header eye toggle — rendered only while at
// least one bot is hidden — reveals them dimmed for right-click → Unhide.
// Hiding is a roster-DISPLAY concern only: mentions, group chats, and the
// name-collision check keep the full roster.

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
        return Promise.resolve({ applied: { ui_meta: true } })
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
  isBotHidden,
  fallbackSelectionAfterHide,
  mergeServerMeta,
  saveBotMeta,
  trackInboundActivity,
  $botMeta,
  $botUnread,
  $selectedBot,
  $lastRoster,
  $showHiddenBots,
  $activityToasts,
  setPluginCtx: value => { pluginCtx = value }
};
`)
  vm.runInNewContext(source, context, { filename: 'plugin.js' })
  if (toastsOn) {
    context.__hide.$activityToasts.set(true)
  }
  return { ...context.__hide, notifications, requests }
}

test('hide: saveBotMeta persists hidden:true locally and ships it in server ui_meta', async () => {
  const t = load()
  const writes = []
  t.setPluginCtx({ storage: { set: (key, value) => writes.push({ key, value }) } })

  await t.saveBotMeta('default', { hidden: true })

  assert.equal(t.$botMeta.get().default.hidden, true)
  assert.equal(writes.at(-1).key, 'bot-meta')
  assert.equal(writes.at(-1).value.default.hidden, true)
  const configure = t.requests.find(r => r.method === 'profiles.configure')
  assert.equal(configure.params.ui_meta['hermes-bots'].hidden, true)
})

test('roster: hidden bots are filtered out; remote-source rows of the same name stay visible', () => {
  const t = load()
  t.$botMeta.set({ ghost: { hidden: true } })
  const roster = [
    { name: 'default' },
    { name: 'ghost' },
    { name: 'ghost', remoteSource: true, connectionId: 'mini' }
  ]
  const meta = t.$botMeta.get()
  const visible = roster.filter(bot => !t.isBotHidden(bot, meta))

  assert.equal(JSON.stringify(visible.map(b => `${b.remoteSource ? 'r:' : ''}${b.name}`)), JSON.stringify(['default', 'r:ghost']))
})

test('unhide: hidden:false clears locally AND survives a mergeServerMeta round-trip', async () => {
  const t = load()
  const writes = []
  t.setPluginCtx({ storage: { set: (key, value) => writes.push({ key, value }) } })
  t.$botMeta.set({ ghost: { hidden: true, title: 'Ghost' } })

  await t.saveBotMeta('ghost', { hidden: false })
  assert.equal(t.$botMeta.get().ghost.hidden, false, 'local merge must clear the flag')
  const configure = t.requests.find(r => r.method === 'profiles.configure')
  assert.equal(configure.params.ui_meta['hermes-bots'].hidden, false, 'server copy carries the literal false')

  // Machine B: its stale local copy still says hidden:true; the server
  // overlay (which merges OVER local) must win with the false.
  t.$botMeta.set({ ghost: { hidden: true, title: 'Ghost' } })
  t.mergeServerMeta([{ name: 'ghost', ui_meta: { 'hermes-bots': { hidden: false, title: 'Ghost' } } }])
  assert.equal(t.$botMeta.get().ghost.hidden, false, 'server false must beat stale local true')
})

test('cross-machine: a hide done elsewhere lands via mergeServerMeta and persists to storage', () => {
  const t = load()
  const writes = []
  t.setPluginCtx({ storage: { set: (key, value) => writes.push({ key, value }) } })
  t.$botMeta.set({ ghost: { title: 'Ghost' } })

  t.mergeServerMeta([{ name: 'ghost', ui_meta: { 'hermes-bots': { hidden: true, title: 'Ghost' } } }])

  assert.equal(t.$botMeta.get().ghost.hidden, true)
  assert.equal(writes.at(-1).value.ghost.hidden, true)
})

test('selection: hiding the selected bot falls back to the first visible bot', () => {
  const t = load()
  t.$selectedBot.set('ghost')
  t.$botMeta.set({ ghost: { hidden: true } })
  t.$lastRoster.set([{ name: 'ghost' }, { name: 'scribe' }, { name: 'default' }])

  t.fallbackSelectionAfterHide('ghost')

  assert.equal(t.$selectedBot.get(), 'scribe')
})

test('selection: falls back to default when nothing else is visible', () => {
  const t = load()
  t.$selectedBot.set('ghost')
  t.$botMeta.set({ ghost: { hidden: true } })
  t.$lastRoster.set([{ name: 'ghost' }])

  t.fallbackSelectionAfterHide('ghost')

  assert.equal(t.$selectedBot.get(), 'default')
})

test('selection: hiding default with nothing else visible keeps the selection (Routines must not chase a ghost)', () => {
  const t = load()
  t.$selectedBot.set('default')
  t.$botMeta.set({ default: { hidden: true } })
  t.$lastRoster.set([{ name: 'default' }])

  t.fallbackSelectionAfterHide('default')

  assert.equal(t.$selectedBot.get(), 'default')
})

test('selection: hiding an unselected bot never moves the selection', () => {
  const t = load()
  t.$selectedBot.set('scribe')
  t.$botMeta.set({ ghost: { hidden: true } })
  t.$lastRoster.set([{ name: 'ghost' }, { name: 'scribe' }])

  t.fallbackSelectionAfterHide('ghost')

  assert.equal(t.$selectedBot.get(), 'scribe')
})

test('activity: a hidden bot accumulates unread silently but never toasts, even with toasts on', () => {
  const t = load({ toastsOn: true })
  t.$botMeta.set({ ghost: { hidden: true } })
  t.$selectedBot.set('default')

  const rosterAt = ts => [{ name: 'ghost', last_session: { last_active: ts, preview: 'Message from writer: hi' } }]
  t.trackInboundActivity(rosterAt(100)) // seeding poll
  t.trackInboundActivity(rosterAt(200)) // activity past the watermark

  assert.equal(t.$botUnread.get().ghost, true, 'unread must accumulate while hidden')
  assert.equal(t.notifications.length, 0, 'hidden bots never toast')
})

// ── source shape ─────────────────────────────────────────────────────────────

test('shape: visible and hidden bots render through separate recoverable sections', () => {
  assert.match(
    pluginSource,
    /const visibleRoster = roster\.filter\(bot => !isBotHidden\(bot, allMeta\)\)/
  )
  assert.match(pluginSource, /const matchingHiddenBots = rowKindFilter === 'groups' \? \[\] : filteredHiddenBots/)
})

test('shape: Hidden section renders only when recovery is relevant', () => {
  assert.match(pluginSource, /const showHiddenSection = hiddenBots\.length > 0/)
  assert.match(pluginSource, /onClick: \(\) => \$showHiddenBots\.set\(!hiddenExpanded\)/)
})

test('shape: revealed hidden rows are flagged and can be restored', () => {
  const botRow = pluginSource.slice(pluginSource.indexOf('function BotRow('), pluginSource.indexOf('// ── model picker'))
  assert.match(botRow, /name: 'eye-closed'/)
  assert.match(botRow, /children: hidden \? 'Unhide' : 'Hide'/)
  assert.match(botRow, /saveBotMeta\(bot, \{ hidden: !hidden \}\)/)
})

test('shape: hiding never filters mentions, group flows, or the meta/activity sweeps', () => {
  // The full-roster consumers keep the FULL roster: mergeServerMeta,
  // avatar pulls, and activity tracking all run on activeSourceRoster —
  // which is derived from the unfiltered roster, not visibleRoster.
  assert.match(pluginSource, /const activeSourceRoster = roster\.filter\(bot => !bot\.remoteSource\)/)
  assert.match(pluginSource, /mergeServerMeta\(activeSourceRoster, data\?\.fetchedAt \|\| 0\)/)
  assert.match(pluginSource, /trackInboundActivity\(roster\)/)
  // Mention resolution never consults the hidden flag.
  const mentions = pluginSource.slice(
    pluginSource.indexOf('function resolveRosterMentions('),
    pluginSource.indexOf('/** Source-qualified identity for a roster row')
  )
  assert.doesNotMatch(mentions, /hidden/i)
})
