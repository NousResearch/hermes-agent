import { QueryClient } from '@tanstack/react-query'
import { act, cleanup, render, waitFor } from '@testing-library/react'
import { useEffect, useRef } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ClientSessionState } from '@/app/types'
import { createClientSessionState } from '@/lib/chat-runtime'
import type { TodoItem } from '@/lib/todos'
import {
  $clarifyRequests,
  clearClarifyRequest,
  setClarifyRequest,
  updateClarifyAnswerDraft
} from '@/store/clarify'
import { clearSessionDraft, stashSessionDraft, takeSessionDraft } from '@/store/composer'
import { $todosBySession, clearSessionTodos, setSessionTodos } from '@/store/todos'
import type { RpcEvent } from '@/types/hermes'

import { useMessageStream } from './index'

const { requestComposerInsert } = vi.hoisted(() => ({ requestComposerInsert: vi.fn() }))

vi.mock('@/app/chat/composer/focus', () => ({ requestComposerInsert }))

const SID = 'session-1'
const todo = (id: string, status: TodoItem['status']): TodoItem => ({ content: `task ${id}`, id, status })

let activeSessionIdRef = { current: SID as string | null }
let handleEvent: ((event: RpcEvent) => void) | null = null

function Harness() {
  const sessionStateByRuntimeIdRef = useRef(new Map<string, ClientSessionState>())
  const queryClientRef = useRef(new QueryClient())

  const stream = useMessageStream({
    activeSessionIdRef,
    hydrateFromStoredSession: vi.fn(async () => undefined),
    queryClient: queryClientRef.current,
    refreshHermesConfig: vi.fn(async () => undefined),
    refreshSessions: vi.fn(async () => undefined),
    sessionStateByRuntimeIdRef,
    updateSessionState: (sessionId, updater) => {
      const current = sessionStateByRuntimeIdRef.current.get(sessionId) ?? createClientSessionState()
      const next = updater(current)
      sessionStateByRuntimeIdRef.current.set(sessionId, next)

      return next
    }
  })

  useEffect(() => {
    handleEvent = stream.handleGatewayEvent
  }, [stream.handleGatewayEvent])

  return null
}

async function mountStream() {
  render(<Harness />)
  await waitFor(() => expect(handleEvent).not.toBeNull())
}

const complete = () => act(() => handleEvent!({ payload: { text: 'done' }, session_id: SID, type: 'message.complete' }))

describe('useMessageStream turn-end todo cleanup', () => {
  beforeEach(() => {
    activeSessionIdRef = { current: SID }
    handleEvent = null
    requestComposerInsert.mockReset()
    $clarifyRequests.set({})
    clearSessionDraft(SID)
    clearSessionDraft('session-2')
    clearSessionTodos(SID)
  })

  afterEach(() => {
    cleanup()
    clearClarifyRequest()
    clearSessionDraft(SID)
    clearSessionDraft('session-2')
    clearSessionTodos(SID)
    vi.restoreAllMocks()
    vi.useRealTimers()
  })

  it('drops a still-active task list when the turn completes', async () => {
    await mountStream()
    setSessionTodos(SID, [todo('a', 'completed'), todo('b', 'in_progress')])

    complete()

    expect($todosBySession.get()[SID]).toBeUndefined()
  })

  it('keeps a finished list on completion so its linger shows the final checkmarks', async () => {
    await mountStream()
    setSessionTodos(SID, [todo('a', 'completed')])

    complete()

    // Not cleared immediately — the finished-list linger still owns it.
    expect($todosBySession.get()[SID]).toHaveLength(1)
  })

  it('drops a still-active task list when the turn errors out', async () => {
    await mountStream()
    setSessionTodos(SID, [todo('a', 'in_progress')])

    act(() => handleEvent!({ payload: { message: 'boom' }, session_id: SID, type: 'error' }))

    expect($todosBySession.get()[SID]).toBeUndefined()
  })

  it('recovers a background clarify draft when that turn completes before submit', async () => {
    await mountStream()
    setClarifyRequest({
      choices: null,
      question: 'Which release notes should I use?',
      requestId: 'clarify-2',
      sessionId: 'session-2'
    })
    updateClarifyAnswerDraft('clarify-2', 'session-2', 'Use the customer-facing release notes.')

    act(() => handleEvent!({ payload: { text: 'done' }, session_id: 'session-2', type: 'message.complete' }))

    expect(takeSessionDraft('session-2').text).toBe(
      'Unsent answer to Hermes question:\n' +
        'Which release notes should I use?\n\n' +
        'Use the customer-facing release notes.'
    )
    expect($clarifyRequests.get()['session-2']).toBeUndefined()
  })

  it('keeps an active clarify draft with its session when the user switches during recovery', async () => {
    await mountStream()
    vi.useFakeTimers()
    setClarifyRequest({
      choices: null,
      question: 'Which release notes should I use?',
      requestId: 'clarify-1',
      sessionId: SID
    })
    updateClarifyAnswerDraft('clarify-1', SID, 'Use the customer-facing release notes.')

    complete()
    // Session-switch cleanup can flush the live composer over the first stash
    // before the delayed recovery callback runs.
    stashSessionDraft(SID, 'existing live draft', [])
    activeSessionIdRef.current = 'session-2'
    await vi.advanceTimersByTimeAsync(100)

    expect(takeSessionDraft(SID).text).toBe(
      'existing live draft\n\n' +
        'Unsent answer to Hermes question:\n' +
        'Which release notes should I use?\n\n' +
        'Use the customer-facing release notes.'
    )
    expect(takeSessionDraft('session-2').text).toBe('')
    expect(requestComposerInsert).not.toHaveBeenCalled()
  })

  it('restores an active clarify draft only after its session is still confirmed', async () => {
    await mountStream()
    vi.useFakeTimers()
    setClarifyRequest({
      choices: null,
      question: 'Which release notes should I use?',
      requestId: 'clarify-1',
      sessionId: SID
    })
    updateClarifyAnswerDraft('clarify-1', SID, 'Use the customer-facing release notes.')

    complete()
    await vi.advanceTimersByTimeAsync(100)

    const recovered =
      'Unsent answer to Hermes question:\n' +
      'Which release notes should I use?\n\n' +
      'Use the customer-facing release notes.'

    expect(takeSessionDraft(SID).text).toBe(recovered)
    expect(requestComposerInsert).toHaveBeenCalledWith(recovered, { mode: 'block', target: 'main' })
  })
})
