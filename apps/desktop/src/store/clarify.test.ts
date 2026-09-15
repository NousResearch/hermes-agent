import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import {
  $clarifyRequest,
  $clarifyRequests,
  type ClarifyRequest,
  clearClarifyRequest,
  displayChoices,
  hasClarifyRequest,
  setClarifyRequest,
  skipClarifyRequest
} from './clarify'
import { $gateway } from './gateway'
import { rememberServerRequest, resetServerRequestsForTests } from './server-requests'
import { $activeSessionId } from './session'

function clarify(sessionId: string | null, requestId: string): ClarifyRequest {
  return {
    choices: null,
    kind: 'single',
    multi_select: false,
    question: `question-${requestId}`,
    requestId,
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

  it('clears by request id across sessions when no session hint is given', () => {
    setClarifyRequest(clarify('session-a', 'shared'))
    setClarifyRequest(clarify('session-b', 'other'))

    clearClarifyRequest('shared')

    expect($clarifyRequests.get()['session-a']).toBeUndefined()
    expect($clarifyRequests.get()['session-b']?.requestId).toBe('other')
  })
})

describe('skipClarifyRequest', () => {
  const request = vi.fn(async () => ({ ok: true }))

  beforeEach(() => {
    $clarifyRequests.set({})
    resetServerRequestsForTests()
    request.mockClear()
    $gateway.set({ request } as unknown as ReturnType<typeof $gateway.get>)
  })

  afterEach(() => {
    $clarifyRequests.set({})
    $gateway.set(null)
  })

  it('answers the session\u2019s clarify with an empty answer and drops it', async () => {
    const respond = vi.fn()

    rememberServerRequest({
      fail: vi.fn(),
      id: 'req-a',
      method: 'clarify',
      params: { choices: null, kind: 'single', multi_select: false, question: 'q', session_id: 'session-a' },
      respond,
      sessionId: 'session-a'
    })
    setClarifyRequest(clarify('session-a', 'req-a'))
    setClarifyRequest(clarify('session-b', 'req-b'))

    await expect(skipClarifyRequest('session-a')).resolves.toBe(true)

    expect(respond).toHaveBeenCalledWith({ answer: '' })
    expect(hasClarifyRequest('session-a')).toBe(false)
    // A background session's question is untouched — only the one being typed
    // over is skipped.
    expect(hasClarifyRequest('session-b')).toBe(true)
  })

  it('is a no-op when the session has no clarify parked', async () => {
    await expect(skipClarifyRequest('session-a')).resolves.toBe(false)
    expect(request).not.toHaveBeenCalled()
  })

  it('still reports the skip when the server request is already gone (expired / other window answered)', async () => {
    setClarifyRequest(clarify('session-a', 'req-a'))

    await expect(skipClarifyRequest('session-a')).resolves.toBe(true)
    expect(hasClarifyRequest('session-a')).toBe(false)
  })
})

describe('displayChoices', () => {
  it('is null when there is nothing to render as a button', () => {
    expect(displayChoices(null)).toBeNull()
    expect(displayChoices([])).toBeNull()
    expect(displayChoices(['', '   '])).toBeNull()
  })

  it('drops blank, multi-line and over-long choices and keeps the rest in order', () => {
    const long = 'x'.repeat(201)
    const ok = 'y'.repeat(200)

    expect(displayChoices(['a', '', 'b\nc', long, ok, '  '])).toEqual(['a', ok])
  })

  it('measures a choice without the recommended label the backend appended', () => {
    const recommended = `${'z'.repeat(200)} (Recommended)`

    expect(displayChoices([recommended, 'plain'])).toEqual([recommended, 'plain'])
  })
})
