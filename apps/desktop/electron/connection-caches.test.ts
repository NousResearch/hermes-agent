import assert from 'node:assert/strict'

import { beforeEach, test } from 'vitest'

import {
  connectionInstallIds,
  evictConnectionCaches,
  rosterSourceErrors,
  sshInventoryAttemptedAt,
  sshInventorySucceededAt,
  sshRosterCache
} from './connection-caches'
import { shouldRetrySshInventory } from './connection-registry'

const MINUTE = 60_000

beforeEach(() => {
  sshRosterCache.clear()
  sshInventoryAttemptedAt.clear()
  sshInventorySucceededAt.clear()
  connectionInstallIds.clear()
  rosterSourceErrors.clear()
})

function seed(id: string) {
  sshRosterCache.set(id, ['default', 'dixie'])
  sshInventoryAttemptedAt.set(id, Date.now())
  sshInventorySucceededAt.set(id, Date.now())
  connectionInstallIds.set(id, { id: 'aaa', ts: Date.now() })
  rosterSourceErrors.set(id, 'previous failure')
}

test('evicting a connection id forgets every cache keyed by it', () => {
  seed('mac-mini')
  seed('spark')

  evictConnectionCaches('mac-mini')

  // Nothing about the evicted id survives in ANY connection-scoped cache. A cache added later
  // and not registered in the module would fail this by omission.
  assert.equal(sshRosterCache.has('mac-mini'), false)
  assert.equal(sshInventoryAttemptedAt.has('mac-mini'), false)
  assert.equal(sshInventorySucceededAt.has('mac-mini'), false)
  assert.equal(connectionInstallIds.has('mac-mini'), false)
  assert.equal(rosterSourceErrors.has('mac-mini'), false)

  // Its neighbours are untouched.
  assert.deepEqual(sshRosterCache.get('spark'), ['default', 'dixie'])
  assert.equal(connectionInstallIds.get('spark')?.id, 'aaa')
  assert.equal(rosterSourceErrors.get('spark'), 'previous failure')
})

test('an evicted id enumerates from the live target again instead of serving the old one', () => {
  // The reason eviction matters: without it a recycled or re-pointed id keeps answering with the
  // previous machine's inventory long after the roster's own TTL would have caught it.
  seed('mac-mini')
  assert.equal(
    shouldRetrySshInventory(sshRosterCache.has('mac-mini'), sshInventoryAttemptedAt.get('mac-mini'), Date.now()),
    false
  )

  evictConnectionCaches('mac-mini')

  assert.equal(
    shouldRetrySshInventory(sshRosterCache.has('mac-mini'), sshInventoryAttemptedAt.get('mac-mini'), Date.now()),
    true
  )
})

test('evicting an unknown or empty id is a no-op', () => {
  seed('mac-mini')

  for (const id of ['', 'never-registered', undefined as unknown as string]) {
    evictConnectionCaches(id)
  }

  assert.equal(sshRosterCache.size, 1)
  assert.equal(sshInventoryAttemptedAt.size, 1)
  assert.equal(sshInventorySucceededAt.size, 1)
  assert.equal(connectionInstallIds.size, 1)
  assert.equal(rosterSourceErrors.size, 1)
})

test('a cached roster is re-read once it is older than the TTL', () => {
  const confirmedAt = 1_000_000

  sshRosterCache.set('pi', ['default'])
  sshInventoryAttemptedAt.set('pi', confirmedAt)
  sshInventorySucceededAt.set('pi', confirmedAt)

  // A profile created on the host is invisible until the cached list is refreshed, so the roster
  // must stop answering once it can no longer describe that host.
  assert.equal(
    shouldRetrySshInventory(true, confirmedAt, confirmedAt + 6 * MINUTE, MINUTE, confirmedAt),
    true
  )

  // The refreshed roster is stamped again, which is what spaces the next refresh a TTL out.
  const refreshedAt = confirmedAt + 6 * MINUTE
  sshInventorySucceededAt.set('pi', refreshedAt)

  assert.equal(
    shouldRetrySshInventory(true, refreshedAt, refreshedAt + 4 * MINUTE, MINUTE, refreshedAt),
    false
  )
})

test('a roster inside the TTL survives the ~5s roster poll without dialling the host again', () => {
  const confirmedAt = 1_000_000

  // What the cached branch used to be: hasCache => false, forever. Kept for every poll that lands
  // inside the TTL, which is all but one poll per interval.
  for (const elapsed of [0, 5_000, 4 * MINUTE]) {
    assert.equal(
      shouldRetrySshInventory(true, confirmedAt, confirmedAt + elapsed, MINUTE, confirmedAt),
      false,
      `polled ${elapsed}ms after the roster was read`
    )
  }
})

test('a stale roster waits for the retry cooldown instead of dialling on every roster poll', () => {
  const confirmedAt = 1_000_000
  const attemptedAt = confirmedAt + 6 * MINUTE

  assert.equal(
    shouldRetrySshInventory(true, attemptedAt, attemptedAt + 5_000, MINUTE, confirmedAt),
    false
  )
  assert.equal(
    shouldRetrySshInventory(true, attemptedAt, attemptedAt + MINUTE, MINUTE, confirmedAt),
    true
  )
})

test('a cache with no read recorded behind it keeps answering instead of being dialled blind', () => {
  // Only the probe stamps a read, so an unstamped cache is one this build did not write: its age is
  // unknowable, and redialing every host on an unknown-age roster would be the worse default.
  assert.equal(
    shouldRetrySshInventory(true, 1_000_000, 1_000_000 + 60 * MINUTE, MINUTE, null),
    false
  )
})

test('a never-read connection still enumerates on first sight and then backs off', () => {
  assert.equal(shouldRetrySshInventory(false, undefined, 1_000_000, MINUTE), true)
  assert.equal(shouldRetrySshInventory(false, 1_000_000, 1_000_000 + 30_000, MINUTE), false)
  assert.equal(shouldRetrySshInventory(false, 1_000_000, 1_000_000 + MINUTE, MINUTE), true)
})
