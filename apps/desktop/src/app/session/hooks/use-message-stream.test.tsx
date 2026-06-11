import { QueryClient } from '@tanstack/react-query'
import { cleanup, render } from '@testing-library/react'
import { useEffect, useRef } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { chatMessageText } from '@/lib/chat-messages'
import type { RpcEvent } from '@/types/hermes'

import type { ClientSessionState } from '../../types'

import { useMessageStream } from './use-message-stream'

vi.mock('@/lib/haptics', () => ({
  triggerHaptic: vi.fn()
}))

const SESSION_ID = 'rt-39558'

function initialState(): ClientSessionState {
  return {
    awaitingResponse: false,
    branch: '',
    busy: false,
    cwd: '',
    interrupted: false,
    messages: [],
    needsInput: false,
    pendingBranchGroup: null,
    sawAssistantPayload: false,
    storedSessionId: 'stored-39558',
    streamId: null
  }
}

interface HarnessHandle {
  handleGatewayEvent: (event: RpcEvent) => void
}

function Harness({
  onReady,
  updateSessionState
}: {
  onReady: (handle: HarnessHandle) => void
  updateSessionState: (
    sessionId: string,
    updater: (state: ClientSessionState) => ClientSessionState,
    storedSessionId?: string | null
  ) => ClientSessionState
}) {
  const activeSessionIdRef = useRef<string | null>(SESSION_ID)
  const queryClientRef = useRef(new QueryClient())

  const stream = useMessageStream({
    activeSessionIdRef,
    hydrateFromStoredSession: async () => undefined,
    queryClient: queryClientRef.current,
    refreshHermesConfig: async () => undefined,
    refreshSessions: async () => undefined,
    updateSessionState
  })

  useEffect(() => {
    onReady({ handleGatewayEvent: stream.handleGatewayEvent })
  }, [onReady, stream.handleGatewayEvent])

  return null
}

describe('useMessageStream', () => {
  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
  })

  it('keeps assistant text streamed before a tool call when final text completes the turn', () => {
    let state = initialState()

    const updateSessionState = vi.fn(
      (
        sessionId: string,
        updater: (state: ClientSessionState) => ClientSessionState,
        _storedSessionId?: string | null
      ) => {
        expect(sessionId).toBe(SESSION_ID)
        state = updater(state)

        return state
      }
    )

    let handle: HarnessHandle | null = null

    render(<Harness onReady={h => (handle = h)} updateSessionState={updateSessionState} />)

    handle!.handleGatewayEvent({ payload: {}, session_id: SESSION_ID, type: 'message.start' } as RpcEvent)
    handle!.handleGatewayEvent({
      payload: { text: 'Planning.' },
      session_id: SESSION_ID,
      type: 'message.delta'
    } as RpcEvent)
    handle!.handleGatewayEvent({
      payload: { args: { command: 'pwd' }, name: 'terminal', tool_id: 'tc-1' },
      session_id: SESSION_ID,
      type: 'tool.start'
    } as RpcEvent)
    handle!.handleGatewayEvent({
      payload: { args: { command: 'pwd' }, name: 'terminal', result: 'ok', tool_id: 'tc-1' },
      session_id: SESSION_ID,
      type: 'tool.complete'
    } as RpcEvent)
    handle!.handleGatewayEvent({
      payload: { text: 'Done.' },
      session_id: SESSION_ID,
      type: 'message.complete'
    } as RpcEvent)

    const assistant = state.messages.find(message => message.role === 'assistant')

    expect(assistant?.pending).toBe(false)
    expect(assistant?.parts.map(part => part.type)).toEqual(['text', 'tool-call', 'text'])
    expect(chatMessageText(assistant!)).toBe('Planning.Done.')
  })

  it('does not duplicate streamed text when completion text contains the full final response', () => {
    let state = initialState()

    const updateSessionState = vi.fn(
      (
        sessionId: string,
        updater: (state: ClientSessionState) => ClientSessionState,
        _storedSessionId?: string | null
      ) => {
        expect(sessionId).toBe(SESSION_ID)
        state = updater(state)

        return state
      }
    )

    let handle: HarnessHandle | null = null

    render(<Harness onReady={h => (handle = h)} updateSessionState={updateSessionState} />)

    handle!.handleGatewayEvent({ payload: {}, session_id: SESSION_ID, type: 'message.start' } as RpcEvent)
    handle!.handleGatewayEvent({
      payload: { text: 'Planning. ' },
      session_id: SESSION_ID,
      type: 'message.delta'
    } as RpcEvent)
    handle!.handleGatewayEvent({
      payload: { args: { command: 'pwd' }, name: 'terminal', tool_id: 'tc-1' },
      session_id: SESSION_ID,
      type: 'tool.start'
    } as RpcEvent)
    handle!.handleGatewayEvent({
      payload: { args: { command: 'pwd' }, name: 'terminal', result: 'ok', tool_id: 'tc-1' },
      session_id: SESSION_ID,
      type: 'tool.complete'
    } as RpcEvent)
    handle!.handleGatewayEvent({
      payload: { text: 'Done.' },
      session_id: SESSION_ID,
      type: 'message.delta'
    } as RpcEvent)
    handle!.handleGatewayEvent({
      payload: { text: 'Planning. Done.' },
      session_id: SESSION_ID,
      type: 'message.complete'
    } as RpcEvent)

    const assistant = state.messages.find(message => message.role === 'assistant')

    expect(assistant?.parts.map(part => part.type)).toEqual(['text', 'tool-call', 'text'])
    expect(chatMessageText(assistant!)).toBe('Planning. Done.')
  })
})
