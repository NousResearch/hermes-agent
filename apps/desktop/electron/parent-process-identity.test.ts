import assert from 'node:assert/strict'

import { test, vi } from 'vitest'

import {
  createParentStartMarkerResolver,
  electronProcessStartMarker,
  parentWatchdogEnv,
  spawnTag
} from './parent-process-identity'

test('electronProcessStartMarker uses Electron creation time only for its own PID', () => {
  assert.equal(electronProcessStartMarker(42, 42, 1_723_456_789_123.75), 'winms:1723456789123')
  assert.equal(electronProcessStartMarker(43, 42, 1_723_456_789_123), null)
})

test('electronProcessStartMarker rejects unavailable or invalid creation times', () => {
  assert.equal(electronProcessStartMarker(42, 42, null), null)
  assert.equal(electronProcessStartMarker(42, 42, Number.NaN), null)
  assert.equal(electronProcessStartMarker(42, 42, 0), null)
})

test('parent marker resolver shares and caches a successful probe', async () => {
  const load = vi.fn(async () => 'winms:1723456789123')
  const resolve = createParentStartMarkerResolver({ load })

  assert.deepEqual(await Promise.all([resolve(), resolve()]), ['winms:1723456789123', 'winms:1723456789123'])
  assert.equal(await resolve(), 'winms:1723456789123')
  assert.equal(load.mock.calls.length, 1)
})

test('parent marker resolver degrades a failed probe and retries later', async () => {
  const failure = new Error('powershell timed out')

  const load = vi
    .fn<() => Promise<string>>()
    .mockRejectedValueOnce(failure)
    .mockResolvedValueOnce('win:638908765432100000')

  const onError = vi.fn()
  const resolve = createParentStartMarkerResolver({ load, onError })

  assert.equal(await resolve(), null)
  assert.equal(await resolve(), 'win:638908765432100000')
  assert.deepEqual(onError.mock.calls, [[failure]])
  assert.equal(load.mock.calls.length, 2)
})

test('parent marker resolver reports one diagnostic for a shared failed probe', async () => {
  const failure = new Error('creation time unavailable')
  const load = vi.fn(async () => Promise.reject(failure))
  const onError = vi.fn()
  const resolve = createParentStartMarkerResolver({ load, onError })

  assert.deepEqual(await Promise.all([resolve(), resolve()]), [null, null])
  assert.deepEqual(onError.mock.calls, [[failure]])
  assert.equal(load.mock.calls.length, 1)
})

test('parentWatchdogEnv emits an exact identity when the marker is available', () => {
  assert.deepEqual(parentWatchdogEnv(42, 'winms:1723456789123', 'nonce-1'), {
    HERMES_PARENT_NONCE: 'nonce-1',
    HERMES_PARENT_PID: '42',
    HERMES_PARENT_START_MARKER: 'winms:1723456789123',
    // Spawn tag mirrored by hermes_cli/process_identity.py — spawner create
    // time in seconds derived from the same winms marker.
    HERMES_SPAWN: 'v1:-:serve:42:1723456789.123'
  })
})

test('parentWatchdogEnv atomically falls back to PID-only identity', () => {
  assert.deepEqual(parentWatchdogEnv(42, null, 'unused-nonce'), {
    HERMES_PARENT_PID: '42',
    // Marker probe failed → spawn tag still present, create time unknown.
    HERMES_SPAWN: 'v1:-:serve:42:-'
  })
  assert.throws(() => parentWatchdogEnv(42, '', 'nonce-1'), /marker and nonce must be non-empty/)
  assert.throws(() => parentWatchdogEnv(42, 'winms:1723456789123', ''), /marker and nonce must be non-empty/)
})

test('spawnTag derives seconds from a winms marker and dashes without one', () => {
  assert.equal(spawnTag(42, 'winms:1723456789123'), 'v1:-:serve:42:1723456789.123')
  assert.equal(spawnTag(42, null), 'v1:-:serve:42:-')
  assert.equal(spawnTag(42, 'win:638908765432100000'), 'v1:-:serve:42:-')
  assert.equal(spawnTag(42, 'winms:garbage'), 'v1:-:serve:42:-')
})
