import assert from 'node:assert/strict'

import { afterEach, beforeEach, expect, test, vi } from 'vitest'

import {
  _deferredEntries,
  OWNER_DEFERRED_MAX_ATTEMPTS,
  OWNER_DEFERRED_RETRY_MS,
  OwnerWriteDeferredMarker,
  parkUntilOwnerAvailable
} from './session-owner-deferred'

beforeEach(() => {
  vi.useFakeTimers()
})

afterEach(() => {
  vi.useRealTimers()
  _deferredEntries.clear()
})

test('a parked write stays parked across still-deferred re-entries, and the bound cannot be reset', async () => {
  const key = 'unread:s1:false'
  const stillDeferred = () => Promise.reject(new OwnerWriteDeferredMarker())

  // The original park: one attempt, one scheduled re-entry.
  expect(parkUntilOwnerAvailable(key, stillDeferred)).toBe('parked')
  expect(_deferredEntries.has(key)).toBe(true)

  // Every fired generation rejects with the marker (the store re-parked on
  // its own re-entry) and the re-park refreshes the SAME entry — so the
  // counter only climbs, never resets.
  for (let reentry = 1; reentry < OWNER_DEFERRED_MAX_ATTEMPTS; reentry += 1) {
    await vi.advanceTimersByTimeAsync(OWNER_DEFERRED_RETRY_MS)
    // Still deferred: the marker kept the entry.
    expect(_deferredEntries.has(key)).toBe(true)
    expect(parkUntilOwnerAvailable(key, stillDeferred)).toBe('parked')
  }

  // The final generation fires; the re-entry at the bound reports the
  // deferral expired, deletes the entry, and schedules nothing further.
  await vi.advanceTimersByTimeAsync(OWNER_DEFERRED_RETRY_MS)
  expect(_deferredEntries.has(key)).toBe(true)
  expect(parkUntilOwnerAvailable(key, stillDeferred)).toBe('expired')
  expect(_deferredEntries.has(key)).toBe(false)
})

test('a landed write settles: the entry goes and nothing further fires', async () => {
  const key = 'pin:s2:true'
  let calls = 0

  const thunk = () => {
    calls += 1

    return Promise.resolve()
  }

  expect(parkUntilOwnerAvailable(key, thunk)).toBe('parked')
  await vi.advanceTimersByTimeAsync(OWNER_DEFERRED_RETRY_MS)

  assert.equal(calls, 1)
  assert.equal(_deferredEntries.has(key), false)
  await vi.advanceTimersByTimeAsync(OWNER_DEFERRED_RETRY_MS)
  assert.equal(calls, 1)
})

test('a genuine failure settles the same way — only the marker keeps an entry', async () => {
  const key = 'unread:s3:true'
  const thunk = () => Promise.reject(new Error('storage_unavailable'))

  expect(parkUntilOwnerAvailable(key, thunk)).toBe('parked')
  await vi.advanceTimersByTimeAsync(OWNER_DEFERRED_RETRY_MS)

  assert.equal(_deferredEntries.has(key), false)
})
