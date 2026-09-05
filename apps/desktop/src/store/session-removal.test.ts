import { afterEach, describe, expect, it } from 'vitest'

import type { SessionInfo } from '@/types/hermes'

import { $sessionResumeRequest, requestSessionResume, setSessions } from './session'
import {
  $removedSessionIds,
  $sessionMutationsInFlight,
  beginSessionMutation,
  endSessionMutation,
  isSessionRemovalPending,
  tombstoneSessions,
  untombstoneSessions
} from './session-removal'

function sessionForOwner(connectionId: string): SessionInfo {
  return {
    connection_id: connectionId,
    ended_at: null,
    id: 'same-id',
    input_tokens: 0,
    is_active: false,
    last_active: 0,
    message_count: 0,
    model: null,
    output_tokens: 0,
    preview: null,
    profile: 'default',
    source: 'desktop',
    started_at: 0,
    title: null,
    tool_call_count: 0
  }
}

afterEach(() => {
  $removedSessionIds.set(new Set())
  $sessionMutationsInFlight.set(new Set())
  $sessionResumeRequest.set(null)
  setSessions([])
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

  it('scopes same-id gateway twins while keeping bare legacy reads fail-closed', () => {
    const gatewayA = { connection_id: 'gateway-a', id: 'same-id', profile: 'default' }
    const gatewayB = { connection_id: 'gateway-b', id: 'same-id', profile: 'default' }

    tombstoneSessions([gatewayB])
    beginSessionMutation([gatewayB])

    expect(isSessionRemovalPending(gatewayB)).toBe(true)
    expect(isSessionRemovalPending(gatewayA)).toBe(false)
    // A caller that cannot prove its owner must not guess between the twins.
    expect(isSessionRemovalPending('same-id')).toBe(true)

    untombstoneSessions([gatewayB])
    endSessionMutation([gatewayB])

    expect(isSessionRemovalPending(gatewayA)).toBe(false)
  })

  it('does not let an owner rollback clear an unrelated legacy tombstone', () => {
    const gatewayB = { connection_id: 'gateway-b', id: 'same-id', profile: 'default' }

    tombstoneSessions(['same-id'])
    tombstoneSessions([gatewayB])
    untombstoneSessions([gatewayB])

    expect(isSessionRemovalPending(gatewayB)).toBe(true)

    // An unscoped legacy producer is the only caller allowed to clear its
    // ambiguous legacy representation.
    untombstoneSessions(['same-id'])
    expect(isSessionRemovalPending(gatewayB)).toBe(false)
  })

  it('routes a bare runtime-gone resume to the surviving same-id owner', () => {
    const gatewayA = sessionForOwner('gateway-a')
    const gatewayB = sessionForOwner('gateway-b')
    setSessions([gatewayB, gatewayA])
    tombstoneSessions([gatewayB])

    requestSessionResume('same-id')

    expect($sessionResumeRequest.get()).toMatchObject({
      ownerRoute: { connectionId: 'gateway-a', profile: 'default' },
      sessionId: 'same-id'
    })
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
