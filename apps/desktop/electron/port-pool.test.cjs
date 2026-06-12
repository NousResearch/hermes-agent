/**
 * Tests for electron/port-pool.cjs.
 *
 * Run with: node --test electron/port-pool.test.cjs
 *
 * PortPool is the in-process reservation that closes the pickPort() TOCTOU
 * window. These cover selection order, skipping reserved/unavailable ports,
 * release/reuse, exhaustion, and async probes — without real sockets.
 */

const test = require('node:test')
const assert = require('node:assert/strict')

const { PortPool } = require('./port-pool.cjs')

const allFree = () => true

test('reserve returns the lowest free port and reserves it', async () => {
  const pool = new PortPool(9120, 9199)
  const port = await pool.reserve(allFree)
  assert.equal(port, 9120)
  assert.ok(pool.has(9120))
  assert.equal(pool.size, 1)
})

test('reserve skips ports already reserved in-process', async () => {
  const pool = new PortPool(9120, 9199)
  const first = await pool.reserve(allFree)
  const second = await pool.reserve(allFree)
  assert.equal(first, 9120)
  assert.equal(second, 9121)
})

test('reserve skips ports the probe rejects', async () => {
  const pool = new PortPool(9120, 9199)
  const busy = new Set([9120, 9121])
  const port = await pool.reserve(p => !busy.has(p))
  assert.equal(port, 9122)
})

test('reserve returns null when every port is taken', async () => {
  const pool = new PortPool(9120, 9121)
  await pool.reserve(allFree)
  await pool.reserve(allFree)
  assert.equal(await pool.reserve(allFree), null)
})

test('release frees a reserved port for reuse', async () => {
  const pool = new PortPool(9120, 9120)
  assert.equal(await pool.reserve(allFree), 9120)
  assert.equal(await pool.reserve(allFree), null) // exhausted
  pool.release(9120)
  assert.ok(!pool.has(9120))
  assert.equal(await pool.reserve(allFree), 9120) // reusable
})

test('release is a no-op for an unreserved port', () => {
  const pool = new PortPool(9120, 9199)
  pool.release(9120)
  assert.equal(pool.size, 0)
})

test('reserve awaits an async probe', async () => {
  const pool = new PortPool(9120, 9199)
  const busy = new Set([9120])
  const port = await pool.reserve(p => Promise.resolve(!busy.has(p)))
  assert.equal(port, 9121)
})

test('clear drops all reservations', async () => {
  const pool = new PortPool(9120, 9199)
  await pool.reserve(allFree)
  await pool.reserve(allFree)
  assert.equal(pool.size, 2)
  pool.clear()
  assert.equal(pool.size, 0)
})
