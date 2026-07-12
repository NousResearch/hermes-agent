import { QueryClient } from '@tanstack/react-query'
import { act, cleanup, render, waitFor } from '@testing-library/react'
import { useEffect, useRef } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ClientSessionState } from '@/app/types'
import { type ChatMessage, chatMessageText } from '@/lib/chat-messages'
import { createClientSessionState } from '@/lib/chat-runtime'
import { FALLBACK_NOTICE_STORAGE_KEY, restoreFallbackNotices } from '@/lib/fallback-notices'
import { $currentFallbackPolicy, setCurrentFallbackPolicy } from '@/store/session'
import type { RpcEvent } from '@/types/hermes'

import { useMessageStream } from './index'

const RUNTIME_ID = 'runtime-1'
const STORED_ID = 'stored-1'

let handleEvent: ((event: RpcEvent) => void) | null = null
let states: Map<string, ClientSessionState>

const message = (id: string, role: ChatMessage['role'], text: string): ChatMessage => ({
  id,
  role,
  parts: [{ type: 'text', text }]
})

function Harness() {
  const activeSessionIdRef = useRef<string | null>(RUNTIME_ID)
  const sessionStateByRuntimeIdRef = useRef(states)
  const queryClientRef = useRef(new QueryClient())

  const stream = useMessageStream({
    activeSessionIdRef,
    hydrateFromStoredSession: vi.fn(async () => undefined),
    queryClient: queryClientRef.current,
    refreshHermesConfig: vi.fn(async () => undefined),
    refreshSessions: vi.fn(async () => undefined),
    sessionStateByRuntimeIdRef,
    updateSessionState: (sessionId, updater, storedSessionId) => {
      const current =
        sessionStateByRuntimeIdRef.current.get(sessionId) ?? createClientSessionState(storedSessionId ?? null)

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

describe('desktop fallback status', () => {
  beforeEach(() => {
    window.localStorage.clear()
    handleEvent = null
    setCurrentFallbackPolicy('')

    const initial = createClientSessionState(STORED_ID, [
      message('user-1', 'user', 'keep going'),
      { ...message('assistant-stream', 'assistant', ''), pending: true }
    ])

    initial.streamId = 'assistant-stream'
    initial.busy = true
    initial.awaitingResponse = true
    states = new Map([[RUNTIME_ID, initial]])
  })

  afterEach(() => {
    cleanup()
    setCurrentFallbackPolicy('')
    vi.restoreAllMocks()
  })

  it('persists a non-terminal decision and restores it before the fallback response', async () => {
    await mountStream()

    act(() =>
      handleEvent!({
        payload: { kind: 'fallback', text: 'Fallback policy local-only: local/a → local/b' },
        session_id: RUNTIME_ID,
        type: 'status.update'
      })
    )

    const live = states.get(RUNTIME_ID)!
    const notice = live.messages.find(item => item.role === 'system')

    expect(notice && chatMessageText(notice)).toBe('Fallback policy local-only: local/a → local/b')
    expect(live.busy).toBe(true)
    expect(live.awaitingResponse).toBe(true)
    expect(window.localStorage.getItem(FALLBACK_NOTICE_STORAGE_KEY)).toContain(STORED_ID)

    const hydrated = restoreFallbackNotices(STORED_ID, [
      message('stored-user', 'user', 'keep going'),
      message('stored-answer', 'assistant', 'done on local/b')
    ])

    expect(hydrated.map(item => item.role)).toEqual(['user', 'system', 'assistant'])
  })

  it('keeps effective policy in the session state reflected by session.info', async () => {
    await mountStream()

    act(() =>
      handleEvent!({
        payload: { fallback_policy: 'local-only' },
        session_id: RUNTIME_ID,
        type: 'session.info'
      })
    )

    expect(states.get(RUNTIME_ID)?.fallbackPolicy).toBe('local-only')
    expect($currentFallbackPolicy.get()).toBe('local-only')
  })
})
