import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import {
  $clarifyRequest,
  $clarifyRequests,
  type ClarifyRequest,
  clearClarifyRequest,
  setClarifyRequest,
  updateClarifyAnswerDraft,
  updateClarifySelectedChoice
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

    expect($clarifyRequests.get()['session-a']?.requestId).toBe('req-a')
    expect($clarifyRequests.get()['session-b']?.requestId).toBe('req-b')
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

  it('preserves a typed answer when the same question refreshes with a new request id', () => {
    const question = 'What context should Hermes use?'
    setClarifyRequest({ ...clarify('session-a', 'req-a1'), question })
    updateClarifyAnswerDraft('req-a1', 'session-a', 'I already wrote the important context.')

    setClarifyRequest({ ...clarify('session-a', 'req-a2'), question })

    expect($clarifyRequests.get()['session-a']).toMatchObject({
      answerDraft: 'I already wrote the important context.',
      requestId: 'req-a2'
    })
  })

  it('keeps picked choices across a remounted matching request', () => {
    setClarifyRequest({ ...clarify('session-a', 'req-a1'), choices: ['A', 'B'] })
    updateClarifySelectedChoice('req-a1', 'session-a', 'B')

    setClarifyRequest({ ...clarify('session-a', 'req-a1'), choices: ['A', 'B'] })

    expect($clarifyRequests.get()['session-a']?.selectedChoice).toBe('B')
  })

  it('drops a picked choice when refreshed choices no longer include it', () => {
    setClarifyRequest({ ...clarify('session-a', 'req-a1'), choices: ['A', 'B'] })
    updateClarifySelectedChoice('req-a1', 'session-a', 'B')

    setClarifyRequest({ ...clarify('session-a', 'req-a1'), choices: ['A', 'C'] })

    expect($clarifyRequests.get()['session-a']?.selectedChoice).toBeNull()
  })

  it('returns cleared requests so callers can recover abandoned drafts', () => {
    setClarifyRequest(clarify('session-a', 'req-a1'))
    updateClarifyAnswerDraft('req-a1', 'session-a', 'do not lose this')

    const cleared = clearClarifyRequest(undefined, 'session-a')

    expect(cleared).toHaveLength(1)
    expect(cleared[0]).toMatchObject({ answerDraft: 'do not lose this', requestId: 'req-a1' })
    expect($clarifyRequests.get()['session-a']).toBeUndefined()
  })

  it('clears by request id across sessions when no session hint is given', () => {
    setClarifyRequest(clarify('session-a', 'shared'))
    setClarifyRequest(clarify('session-b', 'other'))

    clearClarifyRequest('shared')

    expect($clarifyRequests.get()['session-a']).toBeUndefined()
    expect($clarifyRequests.get()['session-b']?.requestId).toBe('other')
  })
})
