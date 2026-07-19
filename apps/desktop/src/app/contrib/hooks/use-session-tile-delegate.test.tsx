import { act, cleanup, render } from '@testing-library/react'
import { type MutableRefObject, useEffect, useRef } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ClientSessionState } from '@/app/types'
import { getSessionMessages } from '@/hermes'
import { textPart } from '@/lib/chat-messages'
import { createClientSessionState } from '@/lib/chat-runtime'
import { $notifications } from '@/store/notifications'
import { $sessions, setSessions } from '@/store/session'
import { sessionTileDelegate, setSessionTileDelegate } from '@/store/session-states'
import type { SessionInfo } from '@/types/hermes'

import { useSessionTileDelegate } from './use-session-tile-delegate'

vi.mock('@/hermes', () => ({
  getSessionMessages: vi.fn(async () => ({ messages: [], session_id: 'session' })),
  PROMPT_SUBMIT_REQUEST_TIMEOUT_MS: 1_800_000,
  setApiRequestProfile: vi.fn()
}))

const STORED_ID = 'stored-tile-peer'
const RUNTIME_ID = 'rt-tile-peer'

function sessionInfo(overrides: Partial<SessionInfo> = {}): SessionInfo {
  return {
    ended_at: null,
    id: STORED_ID,
    input_tokens: 0,
    is_active: true,
    last_active: 0,
    message_count: 2,
    model: null,
    output_tokens: 0,
    preview: null,
    source: null,
    started_at: 0,
    title: 'Tile chat',
    tool_call_count: 0,
    ...overrides
  }
}

function Harness({
  onReady,
  requestGateway,
  sessionStateByRuntimeIdRef,
  updateSessionState
}: {
  onReady: () => void
  requestGateway: <T>(method: string, params?: Record<string, unknown>, timeoutMs?: number) => Promise<T>
  sessionStateByRuntimeIdRef: MutableRefObject<Map<string, ClientSessionState>>
  updateSessionState: (
    sessionId: string,
    updater: (state: ClientSessionState) => ClientSessionState,
    storedSessionId?: string | null
  ) => ClientSessionState
}) {
  const runtimeIdByStoredSessionIdRef = useRef(new Map<string, string>([[STORED_ID, RUNTIME_ID]]))

  useSessionTileDelegate({
    archiveSession: async () => undefined,
    branchStoredSession: async () => undefined,
    executeSlashCommand: async () => undefined,
    removeSession: async () => undefined,
    requestGateway,
    runtimeIdByStoredSessionIdRef,
    sessionStateByRuntimeIdRef,
    updateSessionState
  })

  useEffect(() => {
    onReady()
  }, [onReady])

  return null
}

describe('useSessionTileDelegate stale multi-window guard (#65047)', () => {
  beforeEach(() => {
    vi.mocked(getSessionMessages).mockReset()
    vi.mocked(getSessionMessages).mockImplementation(async () => ({ messages: [], session_id: STORED_ID }))
    $notifications.set([])
    setSessions(() => [sessionInfo({ profile: 'work-vps' })])
    setSessionTileDelegate(null as never)
  })

  afterEach(() => {
    cleanup()
    $notifications.set([])
    $sessions.set([])
    setSessionTileDelegate(null as never)
  })

  it('blocks tile submitToSession when a peer window advanced the transcript', async () => {
    // Resume retains an existing cache (messages.length > 0). A peer Desktop
    // window can finish a turn while this tile still holds the open-time snapshot.
    const staleCache = createClientSessionState(STORED_ID, [
      { id: 'u1', role: 'user', parts: [textPart('a')] },
      { id: 'a1', role: 'assistant', parts: [textPart('b')] }
    ])
    const sessionStateByRuntimeIdRef = { current: new Map([[RUNTIME_ID, staleCache]]) }
    const seeds: ClientSessionState[] = []

    vi.mocked(getSessionMessages).mockResolvedValue({
      session_id: STORED_ID,
      messages: [
        { content: 'a', role: 'user', timestamp: 1 },
        { content: 'b', role: 'assistant', timestamp: 2 },
        { content: 'c', role: 'user', timestamp: 3 },
        { content: 'd', role: 'assistant', timestamp: 4 }
      ]
    })

    const requestGateway = vi.fn(async () => ({}) as never)

    await act(async () => {
      render(
        <Harness
          onReady={() => undefined}
          requestGateway={requestGateway}
          sessionStateByRuntimeIdRef={sessionStateByRuntimeIdRef}
          updateSessionState={(sessionId, updater, storedSessionId) => {
            const prev = sessionStateByRuntimeIdRef.current.get(sessionId) ?? createClientSessionState(storedSessionId)
            const next = updater(prev)
            sessionStateByRuntimeIdRef.current.set(sessionId, next)
            seeds.push(next)

            return next
          }}
        />
      )
    })

    const delegate = sessionTileDelegate()
    expect(delegate).toBeTruthy()

    await act(async () => {
      await delegate!.submitToSession(RUNTIME_ID, 'stale tile send')
    })

    expect(getSessionMessages).toHaveBeenCalledWith(STORED_ID, 'work-vps')
    expect(requestGateway).not.toHaveBeenCalledWith('prompt.submit', expect.anything(), expect.anything())
    expect(seeds.at(-1)?.messages).toHaveLength(4)
    expect(seeds.at(-1)?.busy).toBe(false)
    expect($notifications.get().some(n => n.kind === 'warning')).toBe(true)
  })

  it('allows tile submitToSession when the authoritative transcript is not ahead', async () => {
    const freshCache = createClientSessionState(STORED_ID, [
      { id: 'u1', role: 'user', parts: [textPart('a')] },
      { id: 'a1', role: 'assistant', parts: [textPart('b')] }
    ])
    const sessionStateByRuntimeIdRef = { current: new Map([[RUNTIME_ID, freshCache]]) }

    vi.mocked(getSessionMessages).mockResolvedValue({
      session_id: STORED_ID,
      messages: [
        { content: 'a', role: 'user', timestamp: 1 },
        { content: 'b', role: 'assistant', timestamp: 2 }
      ]
    })

    const requestGateway = vi.fn(async () => ({}) as never)

    await act(async () => {
      render(
        <Harness
          onReady={() => undefined}
          requestGateway={requestGateway}
          sessionStateByRuntimeIdRef={sessionStateByRuntimeIdRef}
          updateSessionState={(sessionId, updater, storedSessionId) => {
            const prev = sessionStateByRuntimeIdRef.current.get(sessionId) ?? createClientSessionState(storedSessionId)
            const next = updater(prev)
            sessionStateByRuntimeIdRef.current.set(sessionId, next)

            return next
          }}
        />
      )
    })

    await act(async () => {
      await sessionTileDelegate()!.submitToSession(RUNTIME_ID, 'fresh tile send')
    })

    expect(getSessionMessages).toHaveBeenCalledWith(STORED_ID, 'work-vps')
    expect(requestGateway).toHaveBeenCalledWith(
      'prompt.submit',
      { session_id: RUNTIME_ID, text: 'fresh tile send' },
      1_800_000
    )
    expect($notifications.get().some(n => n.kind === 'warning')).toBe(false)
  })
})
