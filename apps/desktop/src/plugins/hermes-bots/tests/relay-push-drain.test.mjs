import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'

// Push-notified relay drain (#93091, motivated by #92760 slow replies): the
// gateway broadcasts `bot_relay.outbox.pending` when an envelope lands in the
// outbox; the plugin drains immediately instead of waiting out the 4s poll.
// Contracts:
// - a burst of signals inside RELAY_PUSH_DEBOUNCE_MS collapses to ONE drain;
// - after the debounce fires, a later signal schedules a fresh drain;
// - the subscription is feature-detected (host.onEvent) and disposed, and the
//   4s interval poll remains untouched as the backstop.

const pluginSource = readFileSync(new URL('../plugin.js', import.meta.url), 'utf8')

function loadScheduler() {
  const start = pluginSource.indexOf('function scheduleRelayPushDrain(')
  const end = pluginSource.indexOf('function startBotRelay(', start)
  assert.ok(start > 0, 'plugin defines scheduleRelayPushDrain')
  assert.ok(end > start, 'scheduler has a stable source boundary')

  const context = {
    setTimeout,
    clearTimeout,
    relayDisposed: false,
    relayPushDebounceTimer: null,
    RELAY_PUSH_DEBOUNCE_MS: 50, // shrunk from 250ms to keep the test fast
    drainCalls: 0
  }
  vm.createContext(context)
  vm.runInContext(
    `async function drainRelayOutboxes() { drainCalls += 1 }\n${pluginSource.slice(start, end)}\nglobalThis.scheduleRelayPushDrain = scheduleRelayPushDrain`,
    context
  )
  return context
}

const sleep = ms => new Promise(resolve => setTimeout(resolve, ms))

test('a burst of pending signals collapses to one drain', async () => {
  const ctx = loadScheduler()

  for (let i = 0; i < 6; i += 1) {
    ctx.scheduleRelayPushDrain()
  }

  assert.equal(ctx.drainCalls, 0, 'drain waits for the debounce window')
  await sleep(120)
  assert.equal(ctx.drainCalls, 1, 'six signals inside the window → one drain')
})

test('a signal after the window fires a fresh drain', async () => {
  const ctx = loadScheduler()

  ctx.scheduleRelayPushDrain()
  await sleep(120)
  assert.equal(ctx.drainCalls, 1)

  ctx.scheduleRelayPushDrain()
  ctx.scheduleRelayPushDrain()
  await sleep(120)
  assert.equal(ctx.drainCalls, 2)
})

test('a disposed relay never schedules a drain', async () => {
  const ctx = loadScheduler()
  ctx.relayDisposed = true

  ctx.scheduleRelayPushDrain()
  await sleep(120)
  assert.equal(ctx.drainCalls, 0)
})

test('subscription is feature-detected, disposed, and the poll backstop stays', () => {
  const start = pluginSource.indexOf("host.onEvent('bot_relay.outbox.pending'")
  assert.ok(start > 0, 'plugin subscribes to bot_relay.outbox.pending')
  const block = pluginSource.slice(start - 400, start + 400)
  assert.match(block, /typeof host\.onEvent === 'function'/)
  assert.match(pluginSource, /relayPushUnsub\(\)/)
  // The 4s interval poll is untouched — push is an accelerator, not a
  // replacement (older backends never emit the event).
  assert.match(pluginSource, /RELAY_DRAIN_INTERVAL_MS = 4_000/)
  assert.match(pluginSource, /setInterval\(\(\) => void drainRelayOutboxes\(\), RELAY_DRAIN_INTERVAL_MS\)/)
})

test('a push racing an in-flight drain schedules a follow-up pass instead of dropping', () => {
  // The gateway signature is monotone — one event per new envelope, never
  // re-broadcast — so a push swallowed by the relayDrainBusy early-return
  // would strand its envelope until the poll. The rerun flag re-schedules.
  const drain = pluginSource.slice(
    pluginSource.indexOf('async function drainRelayOutboxes'),
    pluginSource.indexOf('function scheduleRelayPushDrain')
  )
  assert.match(drain, /if \(relayDrainBusy\) \{/)
  assert.match(drain, /relayDrainRerun = true/)
  // finally block: after busy clears, a remembered push re-schedules once.
  const finallyBlock = drain.slice(drain.indexOf('} finally {'))
  assert.match(finallyBlock, /relayDrainBusy = false/)
  assert.match(finallyBlock, /relayDrainRerun && !relayDisposed/)
  assert.match(finallyBlock, /scheduleRelayPushDrain\(\)/)
})
