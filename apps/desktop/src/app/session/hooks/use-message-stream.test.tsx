import { act, cleanup, render, waitFor } from '@testing-library/react'
import type { MutableRefObject } from 'react'
import { useEffect, useRef } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { createClientSessionState } from '@/lib/chat-runtime'
import type { RpcEvent } from '@/types/hermes'

import type { ClientSessionState } from '../../types'

import { useMessageStream } from './use-message-stream'

const RUNTIME_SESSION_ID = 'rt-running-code'
const STORED_SESSION_ID = '20260615_105331_404f55'

interface HarnessHandle {
  handleGatewayEvent: (event: RpcEvent) => void
  stateRef: MutableRefObject<Map<string, ClientSessionState>>
}

function runningWithoutAssistantPayload(): ClientSessionState {
  return {
    ...createClientSessionState(STORED_SESSION_ID),
    awaitingResponse: true,
    busy: true,
    messages: [
      {
        id: 'user-1',
        parts: [{ text: 'count pictures', type: 'text' }],
        role: 'user'
      }
    ],
    sawAssistantPayload: false,
    streamId: 'assistant-stream-1',
    turnStartedAt: 1_720_000_000_000
  }
}

function Harness({
  hydrateFromStoredSession,
  initialState,
  onReady
}: {
  hydrateFromStoredSession: (
    attempts?: number,
    storedSessionId?: null | string,
    runtimeSessionId?: null | string
  ) => Promise<void>
  initialState: ClientSessionState
  onReady: (handle: HarnessHandle) => void
}) {
  const activeSessionIdRef: MutableRefObject<string | null> = { current: RUNTIME_SESSION_ID }
  const stateRef = useRef(new Map([[RUNTIME_SESSION_ID, initialState]]))

  const stream = useMessageStream({
    activeSessionIdRef,
    hydrateFromStoredSession,
    queryClient: { invalidateQueries: vi.fn() } as never,
    refreshHermesConfig: vi.fn(async () => undefined),
    refreshSessions: vi.fn(async () => undefined),
    sessionStateByRuntimeIdRef: stateRef,
    updateSessionState: (sessionId, updater, storedSessionId) => {
      const previous = stateRef.current.get(sessionId) ?? createClientSessionState(storedSessionId ?? null)

      const next = updater({
        ...previous,
        messages: previous.messages,
        storedSessionId: storedSessionId !== undefined ? storedSessionId : previous.storedSessionId
      })

      stateRef.current.set(sessionId, next)

      return next
    }
  })

  useEffect(() => {
    onReady({ handleGatewayEvent: stream.handleGatewayEvent, stateRef })
  }, [onReady, stream.handleGatewayEvent])

  return null
}

describe('useMessageStream session.info running state', () => {
  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
  })

  it('settles and hydrates when running ends before any assistant payload arrives', async () => {
    const hydrateFromStoredSession = vi.fn(async () => undefined)
    let handle: HarnessHandle | null = null

    render(
      <Harness
        hydrateFromStoredSession={hydrateFromStoredSession}
        initialState={runningWithoutAssistantPayload()}
        onReady={value => {
          handle = value
        }}
      />
    )

    await waitFor(() => expect(handle).not.toBeNull())

    act(() => {
      handle!.handleGatewayEvent({
        payload: { running: false },
        session_id: RUNTIME_SESSION_ID,
        type: 'session.info'
      } as RpcEvent)
    })

    const state = handle!.stateRef.current.get(RUNTIME_SESSION_ID)

    expect(state).toMatchObject({
      awaitingResponse: false,
      busy: false,
      needsInput: false,
      pendingBranchGroup: null,
      streamId: null,
      turnStartedAt: null
    })
    expect(hydrateFromStoredSession).toHaveBeenCalledWith(3, STORED_SESSION_ID, RUNTIME_SESSION_ID)
  })
})
