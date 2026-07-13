import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import {
  $clarifyRequest,
  $clarifyRequests,
  type ClarifyRequest,
  clearClarifyRequest,
  setClarifyRequest
} from './clarify'
import { $activeGatewayProfile } from './profile'
import { $activeSessionId } from './session'

function clarify(sessionId: string | null, requestId: string): ClarifyRequest {
  return {
    profile: 'default',
    requestId,
    question: `question-${requestId}`,
    choices: null,
    sessionId
  }
}

describe('clarify store', () => {
  beforeEach(() => {
    clearClarifyRequest()
    $activeGatewayProfile.set('default')
    $activeSessionId.set(null)
  })

  afterEach(() => {
    clearClarifyRequest()
    $activeGatewayProfile.set('default')
    $activeSessionId.set(null)
  })

  it('keeps the same runtime session id independent across normalized profiles', () => {
    setClarifyRequest({ ...clarify('shared', 'default-request'), profile: 'default' })
    setClarifyRequest({ ...clarify('shared', 'work-request'), profile: ' work ' })

    expect(Object.values($clarifyRequests.get()).map(request => request.requestId)).toEqual([
      'default-request',
      'work-request'
    ])
  })

  it('stamps a stable local request identity and receive time at ingest', () => {
    const before = Date.now()

    setClarifyRequest(clarify('runtime', 'backend-request'))

    const stored = Object.values($clarifyRequests.get())[0]

    expect(stored?.receivedAt).toBeGreaterThanOrEqual(before)
    expect(stored?.requestIdentity).toEqual(expect.any(String))
    expect(stored?.requestIdentity).not.toBe('backend-request')
  })

  it('derives and clears requests by exact active profile, session, and request id', () => {
    setClarifyRequest({ ...clarify('shared', 'default-request'), profile: 'default' })
    setClarifyRequest({ ...clarify('shared', 'work-request'), profile: 'work' })
    $activeSessionId.set('shared')
    $activeGatewayProfile.set('work')

    expect($clarifyRequest.get()?.requestId).toBe('work-request')

    clearClarifyRequest({ profile: 'work', requestId: 'stale', sessionId: 'shared' })
    expect($clarifyRequest.get()?.requestId).toBe('work-request')

    clearClarifyRequest({ profile: 'work', requestId: 'work-request', sessionId: 'shared' })
    expect($clarifyRequest.get()).toBeNull()
    expect(Object.values($clarifyRequests.get())).toEqual([
      expect.objectContaining({ profile: 'default', requestId: 'default-request', sessionId: 'shared' })
    ])
  })

  it('keeps clarify requests from concurrent sessions independent', () => {
    setClarifyRequest(clarify('session-a', 'req-a'))
    setClarifyRequest(clarify('session-b', 'req-b'))

    expect(Object.values($clarifyRequests.get()).map(request => request.requestId)).toEqual(['req-a', 'req-b'])
  })

  it('exposes only the active session via the focus-scoped view', () => {
    setClarifyRequest(clarify('session-a', 'req-a'))
    setClarifyRequest(clarify('session-b', 'req-b'))

    $activeSessionId.set('session-a')
    expect($clarifyRequest.get()?.requestId).toBe('req-a')

    $activeSessionId.set('session-b')
    expect($clarifyRequest.get()?.requestId).toBe('req-b')

    $activeSessionId.set('session-c')
    expect($clarifyRequest.get()).toBeNull()
  })

  it('clears only the targeted session, leaving the other pending', () => {
    setClarifyRequest(clarify('session-a', 'req-a'))
    setClarifyRequest(clarify('session-b', 'req-b'))

    clearClarifyRequest({ profile: 'default', requestId: 'req-a', sessionId: 'session-a' })

    expect(Object.values($clarifyRequests.get()).map(request => request.requestId)).toEqual(['req-b'])
  })

  it('ignores a stale clear whose request id no longer matches', () => {
    setClarifyRequest(clarify('session-a', 'req-a2'))

    clearClarifyRequest({ profile: 'default', requestId: 'req-a1', sessionId: 'session-a' })

    expect(Object.values($clarifyRequests.get()).map(request => request.requestId)).toEqual(['req-a2'])
  })

  it('clears by request id across sessions when no session hint is given', () => {
    setClarifyRequest(clarify('session-a', 'shared'))
    setClarifyRequest(clarify('session-b', 'other'))

    clearClarifyRequest('shared')

    expect(Object.values($clarifyRequests.get()).map(request => request.requestId)).toEqual(['other'])
  })
})
