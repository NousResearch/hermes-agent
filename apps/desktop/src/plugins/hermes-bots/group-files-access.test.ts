import { describe, expect, it } from 'vitest'

import {
  beginGroupFileDelivery,
  captureGroupFileAccess,
  confirmGroupFileCatalog,
  groupFileAccessCurrent,
  invalidateGroupFileAccess,
  subscribeGroupFileAccess
} from './group-files-access'
import { FILE_ROOM } from './group-files-test-fixtures'

describe('room-scoped observed-denial fence', () => {
  it('a newer denial invalidates a recovery even when access was already blocked', () => {
    const room = { ...FILE_ROOM, roomId: 'blocked-recovery-order' }
    invalidateGroupFileAccess(captureGroupFileAccess(room))
    const firstRecovery = captureGroupFileAccess(room)
    const secondRecovery = captureGroupFileAccess(room)
    invalidateGroupFileAccess(secondRecovery)
    expect(() => confirmGroupFileCatalog(firstRecovery)).toThrow()
    expect(() => beginGroupFileDelivery(room)).toThrow()
    confirmGroupFileCatalog(captureGroupFileAccess(room))
    const delivery = beginGroupFileDelivery(room)
    expect(delivery.current()).toBe(true)
    delivery.release()
  })

  it('aborts all leases before notifying UI and does not clear on an old catalog receipt', () => {
    const room = { ...FILE_ROOM, roomId: 'fence-order' }
    const first = beginGroupFileDelivery(room)
    const second = beginGroupFileDelivery(room)
    let notified = false

    const unsubscribe = subscribeGroupFileAccess(first.token, () => {
      notified = true
      expect(first.signal.aborted).toBe(true)
      expect(second.signal.aborted).toBe(true)
    })

    invalidateGroupFileAccess(first.token)
    expect(notified).toBe(true)
    expect(() => confirmGroupFileCatalog(first.token)).toThrow()
    expect(() => beginGroupFileDelivery(room)).toThrow()
    const fresh = captureGroupFileAccess(room)
    confirmGroupFileCatalog(fresh)
    expect(groupFileAccessCurrent(fresh)).toBe(true)
    expect(first.current()).toBe(false)
    expect(second.current()).toBe(false)
    const recovered = beginGroupFileDelivery(room)
    invalidateGroupFileAccess(first.token)
    expect(recovered.current()).toBe(true)
    first.release()
    second.release()
    recovered.release()
    unsubscribe()
  })

  it('isolates unrelated rooms and a new authority epoch', () => {
    const room = { ...FILE_ROOM, roomId: 'fence-isolation' }
    const old = beginGroupFileDelivery(room)
    const another = beginGroupFileDelivery({ ...room, roomId: 'another-room' })
    const replacement = beginGroupFileDelivery({ ...room, hostedEpoch: 2 })
    invalidateGroupFileAccess(old.token)
    expect(old.current()).toBe(false)
    expect(another.current()).toBe(true)
    expect(replacement.current()).toBe(true)
    old.release()
    another.release()
    replacement.release()
  })

  it('releases completed delivery controllers instead of recalling already delivered work', () => {
    const room = { ...FILE_ROOM, roomId: 'fence-complete' }
    const completed = beginGroupFileDelivery(room)
    completed.release()
    invalidateGroupFileAccess(completed.token)
    expect(completed.signal.aborted).toBe(false)
  })
})
