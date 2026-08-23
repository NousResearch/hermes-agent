import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'

// The cross-connection bot relay (Aug 2026 ruling: connections ARE the peer
// set). Source-shape contracts:
// - both relay loops exist, start in register(), and stop via ctx.onDispose;
// - the roster loop pushes bot_relay.roster.sync with agents from OTHER
//   connections only;
// - the drain loop wires drain → deliver → reply, and posts an error reply
//   when the target connection is gone (waiter must never dangle);
// - the remote-row toast no longer tells users messaging is device-local;
// - the middleware note carries the message_agent target for cross-
//   connection rows instead of implying they're unreachable.

const pluginSource = readFileSync(new URL('../plugin.js', import.meta.url), 'utf8')

test('relay loops start in register() and stop on dispose', () => {
  assert.match(pluginSource, /startBotRelay\(\)/)
  assert.match(pluginSource, /ctx\.onDispose\(stopBotRelay\)/)
  // teardown really clears both timers
  const stop = pluginSource.slice(pluginSource.indexOf('function stopBotRelay'))
  assert.match(stop.slice(0, 500), /clearInterval\(relayRosterTimer\)/)
  assert.match(stop.slice(0, 500), /clearInterval\(relayDrainTimer\)/)
})

test('roster loop syncs OTHER connections agents to each gateway', () => {
  const sync = pluginSource.slice(
    pluginSource.indexOf('async function syncRelayRosters'),
    pluginSource.indexOf('async function drainRelayOutboxes')
  )
  assert.match(sync, /bot_relay\.roster\.sync/)
  assert.match(sync, /id !== connection\.id/)
})

test('drain loop wires drain → deliver → reply with error fallback', () => {
  const drain = pluginSource.slice(
    pluginSource.indexOf('async function drainRelayOutboxes'),
    pluginSource.indexOf('function startBotRelay')
  )
  assert.match(drain, /bot_relay\.outbox\.drain/)
  assert.match(drain, /bot_relay\.deliver/)
  assert.match(drain, /bot_relay\.reply/)
  // a missing target connection still posts a reply (error) for the waiter
  assert.match(drain, /is not connected to this Desktop right now/)
})

test('remote-row dead-end toast is gone (rows open; relay carries DMs)', () => {
  assert.doesNotMatch(pluginSource, /Gateway stays on this device/)
})

test('middleware note names the cross-connection message_agent target', () => {
  assert.match(pluginSource, /message_agent target: "\$\{target\}"/)
  assert.match(pluginSource, /agents on other connected machines are reachable too/)
})
