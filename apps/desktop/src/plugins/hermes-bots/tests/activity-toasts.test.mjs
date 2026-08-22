import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'

// Activity toasts are OPT-IN: by default new bot activity only sets the
// unread badge — host.notify fires only when the 'activity-toasts' pref is
// enabled. A busy roster (cron runs, bot-to-bot chatter) must not firehose
// the user with notifications out of the box.

const source = readFileSync(new URL('../plugin.js', import.meta.url), 'utf8')

function loadTracker(toastsEnabled, { selectedBot = 'default', selectedRosterKey = '' } = {}) {
  const start = source.indexOf('const rosterWatermarks = new Map()')
  const end = source.indexOf('/** Last good cron list', start)
  // The tracker keys watermarks off the REAL botActivitySession helper
  // (defined later in plugin.js) — extract it so the harness can't drift.
  const helperStart = source.indexOf('function botActivitySession(')
  const helperEnd = source.indexOf('/** Bots that are working', helperStart)
  const keyStart = source.indexOf('function botRosterKey(')
  const keyEnd = source.indexOf('function sourceByConnection(', keyStart)
  assert.ok(helperStart >= 0 && helperEnd > helperStart, 'botActivitySession must remain extractable')
  assert.ok(keyStart >= 0 && keyEnd > keyStart, 'botRosterKey must remain extractable')
  const notifications = []
  const context = {
    pluginCtx: null,
    atom: initial => {
      let value = initial
      return { get: () => value, set: next => { value = next } }
    },
    host: { notify: params => notifications.push(params) },
    $selectedBot: { get: () => selectedBot },
    $selectedRosterKey: (() => {
      let value = selectedRosterKey
      return { get: () => value, set: next => { value = next } }
    })(),
    $botMeta: { get: () => ({}) },
    $botUnread: (() => {
      let value = {}
      return { get: () => value, set: next => { value = next } }
    })(),
    displayName: bot => bot.name,
    isBotHidden: () => false
  }
  const section = source
    .slice(helperStart, helperEnd)
    .concat('\n', source.slice(start, end))
    .concat('\n', source.slice(keyStart, keyEnd))
    .concat('\nglobalThis.__t = { trackInboundActivity, $activityToasts, setActivityToasts, $selectedRosterKey, rosterWatermarks };\n')
  vm.runInNewContext(section, context, { filename: 't.js' })
  if (toastsEnabled) {
    context.__t.$activityToasts.set(true)
  }
  return { ...context.__t, notifications, $botUnread: context.$botUnread }
}

function rosterAt(ts) {
  return [{ name: 'researcher', last_session: { last_active: ts, preview: 'Message from writer: hi' } }]
}

test('default: new activity sets unread badge but never toasts', () => {
  const t = loadTracker(false)
  t.trackInboundActivity(rosterAt(100)) // seeding poll
  t.trackInboundActivity(rosterAt(200)) // activity moved past watermark
  assert.equal(t.$botUnread.get()['legacy::researcher'], true, 'unread badge must still be set')
  assert.equal(t.notifications.length, 0, 'no toast by default')
})

test('opt-in: enabling the pref restores per-activity toasts', () => {
  const t = loadTracker(true)
  t.trackInboundActivity(rosterAt(100))
  t.trackInboundActivity(rosterAt(200))
  assert.equal(t.notifications.length, 1)
  assert.match(t.notifications[0].title, /New message for researcher/)
})

test('pref defaults OFF and persists via ctx.storage under activity-toasts', () => {
  const t = loadTracker(false)
  assert.equal(t.$activityToasts.get(), false, 'default must be off')
  assert.match(
    source.slice(source.indexOf('function setActivityToasts('), source.indexOf('/** Detect new inbound activity')),
    /storage\?\.set\?\.\('activity-toasts', enabled\)/
  )
  assert.match(source, /storage\?\.get\?\.\('activity-toasts'\)/)
})

test('activity in the hidden canonical Bot Chat still badges (the "6d ago" class)', () => {
  // The canonical Bot Chat is hidden from session lists, so last_session
  // never advances when a DM lands there — only canonical_session does.
  const t = loadTracker(false)
  const at = ts => [
    {
      name: 'researcher',
      last_session: { last_active: 100, preview: 'ancient scratch chat' },
      canonical_session: { last_active: ts, preview: 'Message from writer: hi' }
    }
  ]
  t.trackInboundActivity(at(150)) // seeding poll
  t.trackInboundActivity(at(250)) // Bot Chat got a DM; last_session unchanged
  assert.equal(t.$botUnread.get()['legacy::researcher'], true, 'hidden Bot Chat activity must set unread')
})

function sameNamedRoster(localAt, remoteAt) {
  return [
    {
      connectionId: 'local',
      name: 'researcher',
      last_session: { last_active: localAt, preview: 'local activity' }
    },
    {
      connectionId: 'work-vps',
      remoteSource: true,
      name: 'researcher',
      last_session: { last_active: remoteAt, preview: 'remote activity' }
    }
  ]
}

test('same-named bots keep selection, unread, and activity watermarks source-qualified', () => {
  // The local bare-name tracker cannot distinguish this selected local bot
  // from its remote twin. The roster key must own all activity state instead.
  const t = loadTracker(false, {
    selectedBot: 'researcher',
    selectedRosterKey: 'local::researcher'
  })

  t.trackInboundActivity(sameNamedRoster(100, 100)) // seed each source
  t.trackInboundActivity(sameNamedRoster(200, 200))

  assert.equal(t.$botUnread.get()['local::researcher'], undefined, 'selected local owner is already visible')
  assert.equal(t.$botUnread.get()['work-vps::researcher'], true, 'remote twin activity remains unread')
  assert.equal(t.rosterWatermarks.get('local::researcher'), 200, 'local watermark is retained separately')
  assert.equal(t.rosterWatermarks.get('work-vps::researcher'), 200, 'remote watermark is retained separately')

  t.$botUnread.set({})
  t.$selectedRosterKey.set('work-vps::researcher')
  t.trackInboundActivity(sameNamedRoster(300, 300))

  assert.equal(t.$botUnread.get()['local::researcher'], true, 'local twin activity remains unread after remote selection')
  assert.equal(t.$botUnread.get()['work-vps::researcher'], undefined, 'selected remote owner is already visible')
  assert.equal(t.rosterWatermarks.get('local::researcher'), 300)
  assert.equal(t.rosterWatermarks.get('work-vps::researcher'), 300)
})
