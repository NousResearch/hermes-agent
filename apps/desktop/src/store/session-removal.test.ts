import { afterEach, describe, expect, it } from 'vitest'

import { $sessionResumeRequest, requestSessionResume } from './session'
import {
  $removedSessionIds,
  $sessionMutationsInFlight,
  beginSessionMutation,
  captureSessionTombstoneGenerations,
  endSessionMutation,
  isSessionRemovalPending,
  tombstoneLifecycleChanged,
  tombstoneSessions,
  untombstoneSessions
} from './session-removal'

afterEach(() => {
  $removedSessionIds.set(new Set())
  $sessionMutationsInFlight.set(new Set())
  $sessionResumeRequest.set(null)
})

describe('isSessionRemovalPending', () => {
  it('is true for a tombstoned id and for one whose delete RPC is still in flight', () => {
    tombstoneSessions(['gone'])
    beginSessionMutation(['deleting'])

    expect(isSessionRemovalPending('gone')).toBe(true)
    expect(isSessionRemovalPending('deleting')).toBe(true)
    expect(isSessionRemovalPending('alive')).toBe(false)
  })

  it('goes false again when a failed delete rolls the row back', () => {
    tombstoneSessions(['rolled-back'])
    beginSessionMutation(['rolled-back'])
    expect(isSessionRemovalPending('rolled-back')).toBe(true)

    untombstoneSessions(['rolled-back'])
    endSessionMutation(['rolled-back'])

    expect(isSessionRemovalPending('rolled-back')).toBe(false)
  })

  it('ignores blank ids rather than treating them as pending', () => {
    expect(isSessionRemovalPending('')).toBe(false)
    expect(isSessionRemovalPending('   ')).toBe(false)
    expect(isSessionRemovalPending(null)).toBe(false)
  })
})

describe('tombstone generations', () => {
  it('bumps a per-id generation on add and remove, leaving unrelated ids untouched', () => {
    const beforeAdd = captureSessionTombstoneGenerations()

    tombstoneSessions(['sess-1'])

    expect(tombstoneLifecycleChanged(beforeAdd, ['sess-1'])).toBe(true)
    expect(tombstoneLifecycleChanged(beforeAdd, ['unrelated'])).toBe(false)

    const beforeRemove = captureSessionTombstoneGenerations()
    untombstoneSessions(['sess-1'])

    expect(tombstoneLifecycleChanged(beforeRemove, ['sess-1'])).toBe(true)
  })

  it('detects an add → remove ABA cycle even though membership is back to unchanged', () => {
    // The core #85163 race: while a by-id resolve is in flight, the target is
    // archived AND the archive rolls back. Membership (in vs out) is the same
    // before and after, but the request raced a doomed row.
    const before = captureSessionTombstoneGenerations()

    tombstoneSessions(['aba-1'])
    untombstoneSessions(['aba-1'])

    expect($removedSessionIds.get()).toEqual(new Set())
    expect(tombstoneLifecycleChanged(before, ['aba-1'])).toBe(true)
  })

  it('a snapshot taken after the lifecycle settles compares equal again', () => {
    tombstoneSessions(['settled-1'])
    untombstoneSessions(['settled-1'])

    const after = captureSessionTombstoneGenerations()

    expect(tombstoneLifecycleChanged(after, ['settled-1'])).toBe(false)
  })

  it('ignores blank ids rather than inventing generations', () => {
    const before = captureSessionTombstoneGenerations()

    tombstoneSessions([null, '', '   '])

    expect(tombstoneLifecycleChanged(before, [null, '', '   '])).toBe(false)
  })
})

describe('requestSessionResume refuses a doomed session', () => {
  it('queues a resume for a live session', () => {
    requestSessionResume('live-1')

    expect($sessionResumeRequest.get()?.sessionId).toBe('live-1')
  })

  it('drops the request once the id is tombstoned', () => {
    tombstoneSessions(['deleted-1'])

    requestSessionResume('deleted-1')

    expect($sessionResumeRequest.get()).toBeNull()
  })

  it('drops the request while the delete RPC is still in flight', () => {
    beginSessionMutation(['deleting-1'])

    requestSessionResume('deleting-1')

    expect($sessionResumeRequest.get()).toBeNull()
  })

  it('leaves an earlier live request intact instead of clobbering it', () => {
    requestSessionResume('live-1')
    const queued = $sessionResumeRequest.get()

    tombstoneSessions(['deleted-1'])
    requestSessionResume('deleted-1')

    expect($sessionResumeRequest.get()).toBe(queued)
  })
})
