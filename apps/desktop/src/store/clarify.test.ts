import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import {
  $clarifyInputs,
  $clarifyRequest,
  $clarifyRequests,
  clarifyInputKey,
  type ClarifyRequest,
  clearClarifyRequest,
  setClarifyDraft,
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
    $clarifyInputs.set({})
    $clarifyRequests.set({})
    $activeSessionId.set(null)
  })

  afterEach(() => {
    $clarifyInputs.set({})
    $clarifyRequests.set({})
    $activeSessionId.set(null)
  })

  it('keeps clarify requests from concurrent sessions independent', () => {
    setClarifyRequest(clarify('session-a', 'req-a'))
    setClarifyRequest(clarify('session-b', 'req-b'))

    expect($clarifyRequests.get()['session-a']?.requestId).toBe('req-a')
    expect($clarifyRequests.get()['session-b']?.requestId).toBe('req-b')
  })

  it('isolates same-question drafts while concurrent sessions wait for request ids', () => {
    const question = 'Which target should Hermes update?'
    const pendingA = clarifyInputKey('session-a', null, question)
    const pendingB = clarifyInputKey('session-b', null, question)

    setClarifyDraft(pendingA, 'Only update session A.')
    setClarifyDraft(pendingB, 'Only update session B.')

    setClarifyRequest({ ...clarify('session-a', 'req-a'), question })

    const requestA = clarifyInputKey('session-a', 'req-a', question)

    expect($clarifyInputs.get()[requestA]?.draft).toBe('Only update session A.')
    expect($clarifyInputs.get()[pendingA]).toBeUndefined()
    expect($clarifyInputs.get()[pendingB]?.draft).toBe('Only update session B.')

    clearClarifyRequest('req-a', 'session-a')

    expect($clarifyInputs.get()[requestA]).toBeUndefined()
    expect($clarifyInputs.get()[pendingB]?.draft).toBe('Only update session B.')

    setClarifyRequest({ ...clarify('session-b', 'req-b'), question })

    const requestB = clarifyInputKey('session-b', 'req-b', question)

    expect($clarifyInputs.get()[requestB]?.draft).toBe('Only update session B.')
    expect($clarifyInputs.get()[pendingB]).toBeUndefined()
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
    expect($clarifyRequests.get()['session-b']?.requestId).toBe('req-b')
  })

  it('ignores a stale clear whose request id no longer matches', () => {
    setClarifyRequest(clarify('session-a', 'req-a2'))

    clearClarifyRequest('req-a1', 'session-a')

    expect($clarifyRequests.get()['session-a']?.requestId).toBe('req-a2')
  })

  it('clears by request id across sessions when no session hint is given', () => {
    setClarifyRequest(clarify('session-a', 'shared'))
    setClarifyRequest(clarify('session-b', 'other'))

    clearClarifyRequest('shared')

    expect($clarifyRequests.get()['session-a']).toBeUndefined()
    expect($clarifyRequests.get()['session-b']?.requestId).toBe('other')
  })
})
