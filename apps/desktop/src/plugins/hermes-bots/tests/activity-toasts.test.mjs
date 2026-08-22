import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'

// Activity toasts are OPT-IN: by default new bot activity only sets the
// unread badge — host.notify fires only when the 'activity-toasts' pref is
// enabled. A busy roster (cron runs, bot-to-bot chatter) must not firehose
// the user with notifications out of the box.

const source = readFileSync(new URL('../plugin.js', import.meta.url), 'utf8')

function loadTracker(toastsEnabled) {
  const start = source.indexOf('const rosterWatermarks = new Map()')
  const end = source.indexOf('/** Last good cron list', start)
  // The tracker keys watermarks off the REAL botActivitySession helper
  // (defined later in plugin.js) — extract it so the harness can't drift.
  const helperStart = source.indexOf('function botActivitySession(')
  const helperEnd = source.indexOf('/** Bots that are working', helperStart)
  assert.ok(helperStart >= 0 && helperEnd > helperStart, 'botActivitySession must remain extractable')
  const notifications = []
  const context = {
    pluginCtx: null,
    atom: initial => {
      let value = initial
      return { get: () => value, set: next => { value = next } }
    },
    host: { notify: params => notifications.push(params) },
    $selectedBot: { get: () => 'default' },
    $botMeta: { get: () => ({}) },
    $botUnread: (() => {
      let value = {}
      return { get: () => value, set: next => { value = next } }
    })(),
    displayName: bot => bot.name
  }
  const section = source
    .slice(helperStart, helperEnd)
    .concat('\n', source.slice(start, end))
    .concat('\nglobalThis.__t = { trackInboundActivity, $activityToasts, setActivityToasts };\n')
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
  assert.equal(t.$botUnread.get().researcher, true, 'unread badge must still be set')
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
  assert.equal(t.$botUnread.get().researcher, true, 'hidden Bot Chat activity must set unread')
})
