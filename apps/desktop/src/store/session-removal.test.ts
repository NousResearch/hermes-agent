import { afterEach, describe, expect, it } from 'vitest'

import { $sessionResumeRequest, requestSessionResume } from './session'
import {
  $removedSessionIds,
  $sessionMutationsInFlight,
  beginSessionMutation,
  endSessionMutation,
  isSessionRemovalPending,
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
