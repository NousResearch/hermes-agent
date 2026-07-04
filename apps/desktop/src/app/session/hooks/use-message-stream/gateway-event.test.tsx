import { QueryClient } from '@tanstack/react-query'
import { act, renderHook } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { $clarifyRequests, clearClarifyRequest, setClarifyRequest } from '@/store/clarify'
import { $activeSessionId } from '@/store/session'
import type { RpcEvent } from '@/types/hermes'

import type { ClientSessionState } from '../../../types'

import { useGatewayEventHandler } from './gateway-event'

const SESSION_ID = 'session-clarify'

function clientSession(overrides: Partial<ClientSessionState> = {}): ClientSessionState {
  return {
    storedSessionId: null,
    messages: [],
    branch: '',
    cwd: '',
    model: '',
    provider: '',
    reasoningEffort: '',
    serviceTier: '',
    fast: false,
    yolo: false,
    personality: '',
    busy: false,
    awaitingResponse: false,
    streamId: null,
    sawAssistantPayload: false,
    pendingBranchGroup: null,
    interrupted: false,
    needsInput: true,
    turnStartedAt: null,
    ...overrides
  }
}

function renderGatewayEventHandler() {
  const updateSessionState = vi.fn((
    _sessionId: string,
    updater: (state: ClientSessionState) => ClientSessionState
  ) => updater(clientSession()))

  const { result } = renderHook(() =>
    useGatewayEventHandler({
      activeSessionIdRef: { current: SESSION_ID },
      compactedTurnRef: { current: new Set<string>() },
      lastCwdInfoSessionRef: { current: null },
      nativeSubagentSessionsRef: { current: new Set<string>() },
      appendAssistantDelta: vi.fn(),
      appendReasoningDelta: vi.fn(),
      completeAssistantMessage: vi.fn(),
      failAssistantMessage: vi.fn(),
      flushQueuedDeltas: vi.fn(),
      queryClient: new QueryClient(),
      refreshHermesConfig: vi.fn(async () => undefined),
      sessionInterrupted: vi.fn(() => false),
      updateSessionState,
      upsertToolCall: vi.fn()
    })
  )

  return { handleEvent: result.current, updateSessionState }
}

function toolComplete(name: string): RpcEvent {
  return {
    type: 'tool.complete',
    session_id: SESSION_ID,
    payload: {
      tool_id: 'tool-1',
      name,
      result: {}
    }
  } as RpcEvent
}

afterEach(() => {
  clearClarifyRequest()
  $clarifyRequests.set({})
  $activeSessionId.set(null)
  vi.restoreAllMocks()
})

describe('useGatewayEventHandler clarify completion handling', () => {
  it('clears the pending clarify request when the clarify tool completes', () => {
    $activeSessionId.set(SESSION_ID)
    setClarifyRequest({
      requestId: 'req-clarify',
      sessionId: SESSION_ID,
      question: 'Pick one?',
      choices: ['A', 'B']
    })

    const { handleEvent } = renderGatewayEventHandler()

    act(() => handleEvent(toolComplete('clarify')))

    expect($clarifyRequests.get()[SESSION_ID]).toBeUndefined()
  })

  it('leaves pending clarify requests alone when another tool completes', () => {
    $activeSessionId.set(SESSION_ID)
    setClarifyRequest({
      requestId: 'req-clarify',
      sessionId: SESSION_ID,
      question: 'Pick one?',
      choices: ['A', 'B']
    })

    const { handleEvent } = renderGatewayEventHandler()

    act(() => handleEvent(toolComplete('terminal')))

    expect($clarifyRequests.get()[SESSION_ID]?.requestId).toBe('req-clarify')
  })
})
