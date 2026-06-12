import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import {
  $clarifyRequest,
  $clarifyRequests,
  type ClarifyRequest,
  clearClarifyRequest,
  setClarifyRequest
} from './clarify'
import { $activeSessionId } from './session'

function clarify(sessionId: string | null, requestId: string): ClarifyRequest {
  return {
    requestId,
    question: `question-${requestId}`,
    choices: null,
    sessionId
  }
}

describe('clarify store', () => {
  beforeEach(() => {
    $clarifyRequests.set({})
    $activeSessionId.set(null)
  })

  afterEach(() => {
    $clarifyRequests.set({})
    $activeSessionId.set(null)
  })

  it('keeps clarify requests from concurrent sessions independent', () => {
    setClarifyRequest(clarify('session-a', 'req-a'))
    setClarifyRequest(clarify('session-b', 'req-b'))

    expect($clarifyRequests.get()['session-a']?.[0]?.requestId).toBe('req-a')
    expect($clarifyRequests.get()['session-b']?.[0]?.requestId).toBe('req-b')
  })

  it('queues multiple clarify requests for the same session oldest-first', () => {
    setClarifyRequest(clarify('session-a', 'req-a1'))
    setClarifyRequest(clarify('session-a', 'req-a2'))

    expect($clarifyRequests.get()['session-a']?.map(request => request.requestId)).toEqual(['req-a1', 'req-a2'])
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

    clearClarifyRequest('req-a', 'session-a')

    expect($clarifyRequests.get()['session-a']).toBeUndefined()
    expect($clarifyRequests.get()['session-b']?.[0]?.requestId).toBe('req-b')
  })

  it('clears only the matching request inside a same-session queue', () => {
    setClarifyRequest(clarify('session-a', 'req-a1'))
    setClarifyRequest(clarify('session-a', 'req-a2'))
    $activeSessionId.set('session-a')

    clearClarifyRequest('req-a1', 'session-a')

    expect($clarifyRequests.get()['session-a']?.map(request => request.requestId)).toEqual(['req-a2'])
    expect($clarifyRequest.get()?.requestId).toBe('req-a2')
  })

  it('ignores a stale clear whose request id no longer matches', () => {
    setClarifyRequest(clarify('session-a', 'req-a2'))

    clearClarifyRequest('req-a1', 'session-a')

    expect($clarifyRequests.get()['session-a']?.[0]?.requestId).toBe('req-a2')
  })

  it('clears by request id across sessions when no session hint is given', () => {
    setClarifyRequest(clarify('session-a', 'shared'))
    setClarifyRequest(clarify('session-b', 'other'))

    clearClarifyRequest('shared')

    expect($clarifyRequests.get()['session-a']).toBeUndefined()
    expect($clarifyRequests.get()['session-b']?.[0]?.requestId).toBe('other')
  })

  it('clears the whole session queue when a turn ends', () => {
    setClarifyRequest(clarify('session-a', 'req-a1'))
    setClarifyRequest(clarify('session-a', 'req-a2'))
    $activeSessionId.set('session-a')

    clearClarifyRequest(undefined, 'session-a')

    expect($clarifyRequests.get()['session-a']).toBeUndefined()
    expect($clarifyRequest.get()).toBeNull()
  })
})
