import { QueryClient } from '@tanstack/react-query'
import { act, cleanup, render, waitFor } from '@testing-library/react'
import { useEffect, useRef } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ClientSessionState } from '@/app/types'
import { createClientSessionState } from '@/lib/chat-runtime'
import type { RpcEvent } from '@/types/hermes'

import { useMessageStream } from './index'

const SID = 'session-process-result'
const VISIBLE_TEXT =
  '[IMPORTANT: Background process abc123 completed normally]\nexit_code=0\n--- OUTPUT ---\nok'

let handleEvent: ((event: RpcEvent) => void) | null = null
let sessionStateByRuntimeIdRef: { current: Map<string, ClientSessionState> }
const refreshBackgroundProcesses = vi.fn(async (_sid: string) => undefined)

vi.mock('@/store/composer-status', async importOriginal => {
  const actual = await importOriginal<typeof import('@/store/composer-status')>()
  return {
    ...actual,
    refreshBackgroundProcesses: (sid: string) => refreshBackgroundProcesses(sid)
  }
})

function Harness() {
  const activeSessionIdRef = useRef<string | null>(SID)
  const queryClientRef = useRef(new QueryClient())
  sessionStateByRuntimeIdRef = useRef(new Map<string, ClientSessionState>())

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

function emit(type: RpcEvent['type'], payload: RpcEvent['payload'] = {}) {
  act(() => handleEvent!({ payload, session_id: SID, type }))
}

function systemMessages() {
  const state = sessionStateByRuntimeIdRef.current.get(SID)
  return (state?.messages ?? []).filter(m => m.role === 'system')
}

describe('process/async-delegation result visibility (#64094)', () => {
  beforeEach(() => {
    handleEvent = null
    refreshBackgroundProcesses.mockClear()
  })

  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
  })

  it('does not append a transcript row on status.update(kind=process)', async () => {
    await mountStream()

    emit('status.update', { kind: 'process', text: VISIBLE_TEXT })

    expect(refreshBackgroundProcesses).toHaveBeenCalledWith(SID)
    expect(systemMessages()).toHaveLength(0)
  })

  it('appends exactly one durable system row for paired status.update + review.summary', async () => {
    await mountStream()

    // tui_gateway emits both events for user-visible outcomes. Desktop must
    // only paint review.summary into the transcript (status.update refreshes
    // the process stack only) — otherwise the same text double-paints.
    emit('status.update', { kind: 'process', text: VISIBLE_TEXT })
    emit('review.summary', { text: VISIBLE_TEXT })

    expect(refreshBackgroundProcesses).toHaveBeenCalledWith(SID)
    const rows = systemMessages()
    expect(rows).toHaveLength(1)
    const parts = rows[0]?.parts ?? []
    const text = parts
      .map(p => ('text' in p ? String(p.text) : ''))
      .join('')
    expect(text).toContain('[IMPORTANT: Background process abc123 completed normally]')
  })
})
