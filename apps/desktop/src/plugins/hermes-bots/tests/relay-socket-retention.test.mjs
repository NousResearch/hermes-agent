import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'

// Relay-route socket retention (#93594): while the bot relay is active, each
// registered connection's pooled socket is pinned open via
// host.retainProfileSocket so the drain loop reuses ONE persistent WebSocket
// instead of dialing and tearing down a fresh one per tick. Contracts:
// - retention is feature-detected (host.retainProfileSocket) — older shells
//   simply keep the per-call lease behavior;
// - a connection is pinned once, not once per drain tick;
// - a connection leaving the registry releases exactly its own pin;
// - releaseRelayRetention drops every pin, and stopBotRelay calls it.

const pluginSource = readFileSync(new URL('../plugin.js', import.meta.url), 'utf8')

function loadRetention({ retainProfileSocket } = {}) {
  const start = pluginSource.indexOf('const relayRouteRetentions = new Map()')
  const end = pluginSource.indexOf('/** One representative route per reachable connection id. */', start)
  assert.ok(start > 0, 'plugin defines relayRouteRetentions')
  assert.ok(end > start, 'retention block has a stable source boundary')

  const releases = []
  const context = {
    relayDisposed: false,
    retainCalls: [],
    releases,
    host: {
      retainProfileSocket:
        retainProfileSocket === null
          ? undefined
          : route => {
              context.retainCalls.push(route)
              let released = false
              const release = () => {
                if (!released) {
                  released = true
                  releases.push(route)
                }
              }
              return release
            }
    }
  }
  vm.createContext(context)
  vm.runInContext(
    `${pluginSource.slice(start, end)}
globalThis.syncRelayRetention = syncRelayRetention
globalThis.releaseRelayRetention = releaseRelayRetention
globalThis.relayRouteRetentions = relayRouteRetentions`,
    context
  )
  return context
}

const conn = id => ({ id, route: { connectionId: id, profile: 'default', targetProfile: 'default' } })

test('each connection is pinned ONCE across multiple drain ticks', () => {
  const ctx = loadRetention()
  const connections = [conn('a'), conn('b')]

  for (let tick = 0; tick < 4; tick += 1) {
    ctx.syncRelayRetention(connections)
  }

  assert.equal(ctx.retainCalls.length, 2, 'one pin per connection, not per tick')
  assert.equal(ctx.relayRouteRetentions.size, 2)
  assert.equal(ctx.releases.length, 0, 'nothing released while connections stay registered')
})

test('a connection leaving the registry releases exactly its own pin', () => {
  const ctx = loadRetention()

  ctx.syncRelayRetention([conn('a'), conn('b')])
  ctx.syncRelayRetention([conn('a')])

  assert.equal(ctx.releases.length, 1)
  assert.equal(ctx.releases[0].connectionId, 'b')
  assert.equal(ctx.relayRouteRetentions.size, 1)
})

test('releaseRelayRetention drops every pin (stop/dispose path)', () => {
  const ctx = loadRetention()

  ctx.syncRelayRetention([conn('a'), conn('b'), conn('c')])
  ctx.releaseRelayRetention()

  assert.equal(ctx.releases.length, 3)
  assert.equal(ctx.relayRouteRetentions.size, 0)

  // Idempotent — a second stop releases nothing new.
  ctx.releaseRelayRetention()
  assert.equal(ctx.releases.length, 3)
})

test('a disposed relay never pins new connections but still releases stale ones', () => {
  const ctx = loadRetention()

  ctx.syncRelayRetention([conn('a')])
  ctx.relayDisposed = true
  ctx.syncRelayRetention([conn('b')])

  assert.equal(ctx.retainCalls.length, 1, 'no new pin after dispose')
  assert.equal(ctx.releases.length, 1, "the departed connection's pin was released")
})

test('retention is feature-detected: an older shell without the door is a no-op', () => {
  const ctx = loadRetention({ retainProfileSocket: null })

  ctx.syncRelayRetention([conn('a'), conn('b')])

  assert.equal(ctx.relayRouteRetentions.size, 0)
})

test('stopBotRelay releases retention (source contract)', () => {
  const stop = pluginSource.slice(
    pluginSource.indexOf('function stopBotRelay('),
    pluginSource.indexOf('/** Per-bot appearance')
  )
  assert.match(stop, /releaseRelayRetention\(\)/)

  // And the drain loop reconciles retention with the CURRENT connection set
  // before deciding whether there is anything to relay.
  const drain = pluginSource.slice(
    pluginSource.indexOf('async function drainRelayOutboxes'),
    pluginSource.indexOf('function scheduleRelayPushDrain')
  )
  assert.match(drain, /syncRelayRetention\(/)
})
