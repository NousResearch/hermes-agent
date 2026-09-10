import { describe, expect, it } from 'vitest'

import { createSshSpawnBatchCoordinator } from './ssh-spawn-batch'

const FIRST_BATCH = 'a'.repeat(32)
const SECOND_BATCH = 'b'.repeat(32)

describe('SSH spawn batch coordinator', () => {
  it('shares one opaque batch identity only between scopes of the same registry connection', () => {
    const ids = [FIRST_BATCH, SECOND_BATCH]
    const batches = createSshSpawnBatchCoordinator({ createId: () => ids.shift()! })

    const primary = batches.acquire('', 'connection-one')
    const sibling = batches.acquire('registry:one:writer', 'connection-one')
    const otherConnection = batches.acquire('registry:two:default', 'connection-two')

    expect(primary).toBe(FIRST_BATCH)
    expect(sibling).toBe(FIRST_BATCH)
    expect(otherConnection).toBe(SECOND_BATCH)
  })

  it('retires the batch identity only after its final scoped backend is released', () => {
    const ids = [FIRST_BATCH, SECOND_BATCH]
    const batches = createSshSpawnBatchCoordinator({ createId: () => ids.shift()! })

    batches.acquire('registry:one:default', 'connection-one')
    batches.acquire('registry:one:writer', 'connection-one')
    batches.release('registry:one:default')

    expect(batches.acquire('registry:one:reader', 'connection-one')).toBe(FIRST_BATCH)

    batches.release('registry:one:writer')
    batches.release('registry:one:reader')

    expect(batches.acquire('registry:one:next', 'connection-one')).toBe(SECOND_BATCH)
  })

  it('does not create a batch for legacy SSH routes without a registry identity', () => {
    const batches = createSshSpawnBatchCoordinator({ createId: () => FIRST_BATCH })

    expect(batches.acquire('legacy:default', '')).toBeNull()
    expect(batches.acquire('legacy:writer', undefined)).toBeNull()
  })
})
