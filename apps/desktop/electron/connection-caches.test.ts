import assert from 'node:assert/strict'

import { beforeEach, test } from 'vitest'

import {
  connectionInstallIds,
  evictConnectionCaches,
  sshInventoryAttemptedAt,
  sshRosterCache
} from './connection-caches'
import { shouldRetrySshInventory } from './connection-registry'

beforeEach(() => {
  sshRosterCache.clear()
  sshInventoryAttemptedAt.clear()
  connectionInstallIds.clear()
})

function seed(id: string) {
  sshRosterCache.set(id, ['default', 'dixie'])
  sshInventoryAttemptedAt.set(id, Date.now())
  connectionInstallIds.set(id, { id: 'aaa', ts: Date.now() })
}

test('evicting a connection id forgets every cache keyed by it', () => {
  seed('mac-mini')
  seed('spark')

  evictConnectionCaches('mac-mini')

  // Nothing about the evicted id survives in ANY connection-scoped cache. A cache added later
  // and not registered in the module would fail this by omission.
  assert.equal(sshRosterCache.has('mac-mini'), false)
  assert.equal(sshInventoryAttemptedAt.has('mac-mini'), false)
  assert.equal(connectionInstallIds.has('mac-mini'), false)

  // Its neighbours are untouched.
  assert.deepEqual(sshRosterCache.get('spark'), ['default', 'dixie'])
  assert.equal(connectionInstallIds.get('spark')?.id, 'aaa')
})

test('an evicted id enumerates from the live target again instead of serving the old one', () => {
  // The reason eviction matters: a cached success is never retried, so without it a recycled or
  // re-pointed id keeps answering with the previous machine's inventory for the whole session.
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
  assert.equal(connectionInstallIds.size, 1)
})
