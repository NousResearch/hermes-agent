import { QueryClient } from '@tanstack/react-query'
import { act, cleanup, render, waitFor } from '@testing-library/react'
import { useEffect, useRef } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ClientSessionState } from '@/app/types'
import { chatMessageText } from '@/lib/chat-messages'
import { createClientSessionState } from '@/lib/chat-runtime'
import { $approvalRequest, clearAllPrompts, setApprovalRequest } from '@/store/prompts'
import { $activeSessionId } from '@/store/session'
import { $todosBySession, clearSessionTodos, setSessionTodos } from '@/store/todos'
import type { RpcEvent } from '@/types/hermes'

import { useMessageStream } from './index'

const sideEffects = vi.hoisted(() => ({
  broadcastSessionsChanged: vi.fn(),
  dispatchNativeNotification: vi.fn(),
  flashPetActivity: vi.fn(),
  markPetUnread: vi.fn(),
  playCompletionSound: vi.fn(),
  setPetActivity: vi.fn()
}))

vi.mock('@/lib/completion-sound', () => ({ playCompletionSound: sideEffects.playCompletionSound }))
vi.mock('@/store/native-notifications', () => ({
  dispatchNativeNotification: sideEffects.dispatchNativeNotification
}))
vi.mock('@/store/pet', () => ({
  flashPetActivity: sideEffects.flashPetActivity,
  markPetUnread: sideEffects.markPetUnread,
  setPetActivity: sideEffects.setPetActivity
}))
vi.mock('@/store/session-sync', () => ({ broadcastSessionsChanged: sideEffects.broadcastSessionsChanged }))

const SID = 'session-1'
const BRANCH_GROUP = 'branch-group-1'

let handleEvent: ((event: RpcEvent) => void) | null = null
let currentState: ClientSessionState | null = null
const hydrateFromStoredSession = vi.fn(async () => undefined)
const refreshSessions = vi.fn(async () => undefined)

function Harness() {
  const activeSessionIdRef = useRef<string | null>(SID)

  const sessionStateByRuntimeIdRef = useRef(
    new Map<string, ClientSessionState>([[SID, { ...createClientSessionState(), pendingBranchGroup: BRANCH_GROUP }]])
  )

  const queryClientRef = useRef(new QueryClient())

  const stream = useMessageStream({
    activeSessionIdRef,
    hydrateFromStoredSession,
    queryClient: queryClientRef.current,
    refreshHermesConfig: vi.fn(async () => undefined),
    refreshSessions,
    sessionStateByRuntimeIdRef,
    updateSessionState: (sessionId, updater) => {
      const current = sessionStateByRuntimeIdRef.current.get(sessionId) ?? createClientSessionState()
      const next = updater(current)
      sessionStateByRuntimeIdRef.current.set(sessionId, next)
      currentState = next

      return next
    }
  })

  useEffect(() => {
    handleEvent = stream.handleGatewayEvent
    currentState = sessionStateByRuntimeIdRef.current.get(SID) ?? null
  }, [stream.handleGatewayEvent, sessionStateByRuntimeIdRef])

  return null
}

async function mountStream() {
  render(<Harness />)
  await waitFor(() => expect(handleEvent).not.toBeNull())
}

const emit = (type: string, payload?: Record<string, unknown>) =>
  act(() => handleEvent!({ payload, session_id: SID, type }))

const assistantMessages = () => currentState!.messages.filter(message => message.role === 'assistant')

const expectCompletionSideEffects = (count: number) => {
  expect(sideEffects.playCompletionSound).toHaveBeenCalledTimes(count)
  expect(sideEffects.flashPetActivity).toHaveBeenCalledTimes(count)
  expect(sideEffects.dispatchNativeNotification).toHaveBeenCalledTimes(count)
  expect(sideEffects.broadcastSessionsChanged).toHaveBeenCalledTimes(count)
  expect(refreshSessions).toHaveBeenCalledTimes(count)
}

describe('useMessageStream message.interim', () => {
  beforeEach(() => {
    handleEvent = null
    currentState = null
    $activeSessionId.set(SID)
    clearAllPrompts()
    clearSessionTodos(SID)
    setApprovalRequest({ command: 'echo ok', description: 'test prompt', sessionId: SID })
    setSessionTodos(SID, [{ content: 'verify result', id: 'verify', status: 'in_progress' }])
    vi.spyOn(Date, 'now').mockReturnValue(1_700_000_000_000)
  })

  afterEach(() => {
    cleanup()
    clearAllPrompts()
    clearSessionTodos(SID)
    $activeSessionId.set(null)
    hydrateFromStoredSession.mockClear()
    refreshSessions.mockClear()
    vi.restoreAllMocks()
    Object.values(sideEffects).forEach(mock => mock.mockClear())
  })

  it('finalizes an interim segment without settling the turn, then completes a distinct reply', async () => {
    await mountStream()

    emit('message.start')
    const turnStartedAt = currentState!.turnStartedAt
    emit('message.delta', { text: 'full attempted reply' })

    // A malformed boundary without authoritative text must not erase or
    // finalize the streamed draft.
    emit('message.interim')
    expect(assistantMessages()).toHaveLength(1)
    expect(chatMessageText(assistantMessages()[0]!)).toBe('full attempted reply')
    expect(assistantMessages()[0]!.pending).toBe(true)
    expect(currentState!.streamId).not.toBeNull()

    emit('message.interim', { already_streamed: false, text: 'full attempted reply' })

    expect(assistantMessages()).toHaveLength(1)
    expect(chatMessageText(assistantMessages()[0]!)).toBe('full attempted reply')
    expect(assistantMessages()[0]!.pending).toBe(false)
    const interimId = assistantMessages()[0]!.id

    expect(currentState).toMatchObject({
      awaitingResponse: false,
      busy: true,
      interimBoundaryPending: true,
      pendingBranchGroup: BRANCH_GROUP,
      streamId: null,
      turnStartedAt
    })
    expect($approvalRequest.get()).not.toBeNull()
    expect($todosBySession.get()[SID]).toHaveLength(1)
    expectCompletionSideEffects(0)
    expect(hydrateFromStoredSession).not.toHaveBeenCalled()

    emit('message.delta', { text: 'verified final reply' })
    emit('message.complete', { text: 'verified final reply' })

    expect(assistantMessages()).toHaveLength(2)
    expect(assistantMessages().map(chatMessageText)).toEqual(['full attempted reply', 'verified final reply'])
    expect(assistantMessages().map(message => message.pending ?? false)).toEqual([false, false])
    expect(assistantMessages()[1]!.id).not.toBe(interimId)
    expect(currentState).toMatchObject({
      awaitingResponse: false,
      busy: false,
      interimBoundaryPending: false,
      pendingBranchGroup: null,
      streamId: null,
      turnStartedAt: null
    })
    expectCompletionSideEffects(1)
    expect(hydrateFromStoredSession).not.toHaveBeenCalled()
  })

  it('keeps an identical final completion distinct from an already-streamed interim reply', async () => {
    await mountStream()

    emit('message.start')
    emit('message.delta', { text: 'same reply' })
    emit('message.interim', { already_streamed: true, text: 'same reply' })

    expect(assistantMessages()).toHaveLength(1)
    expect(chatMessageText(assistantMessages()[0]!)).toBe('same reply')
    expect(assistantMessages()[0]!.pending).toBe(false)
    expect(currentState!.interimBoundaryPending).toBe(true)
    const interimId = assistantMessages()[0]!.id

    emit('message.complete', { text: 'same reply' })

    expect(assistantMessages()).toHaveLength(2)
    expect(assistantMessages().map(chatMessageText)).toEqual(['same reply', 'same reply'])
    expect(assistantMessages()[1]!.id).not.toBe(interimId)
    expect(currentState).toMatchObject({ busy: false, interimBoundaryPending: false, streamId: null })
    expectCompletionSideEffects(1)
    expect(hydrateFromStoredSession).not.toHaveBeenCalled()
  })
})
