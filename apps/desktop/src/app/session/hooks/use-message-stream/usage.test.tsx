import { QueryClient } from '@tanstack/react-query'
import { act, cleanup, render, waitFor } from '@testing-library/react'
import { useEffect, useRef } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ClientSessionState } from '@/app/types'
import { createClientSessionState } from '@/lib/chat-runtime'
import { $currentUsage } from '@/store/session'
import type { RpcEvent } from '@/types/hermes'

import { useMessageStream } from './index'

const SID = 'session-1'
// $currentUsage mirrors the primary session; ClientSessionState.usage drives
// the same status bar when a secondary tile is focused.
const BASELINE = { calls: 2, input: 500, output: 40, total: 540 }

let handleEvent: ((event: RpcEvent) => void) | null = null
let sessionStates = new Map<string, ClientSessionState>()

function Harness() {
  const activeSessionIdRef = useRef<string | null>(SID)
  const sessionStateByRuntimeIdRef = useRef(sessionStates)
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

describe('useMessageStream status-bar usage scoping', () => {
  beforeEach(() => {
    handleEvent = null
    sessionStates = new Map([[SID, { ...createClientSessionState(), usage: { ...BASELINE } }]])
    $currentUsage.set({ ...BASELINE })
  })

  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
  })

  it('merges a live session.usage tick from the focused session', async () => {
    await mountStream()

    act(() =>
      handleEvent!({
        payload: { usage: { context_percent: 42, input: 1200, total: 1280 } },
        session_id: SID,
        type: 'session.usage'
      })
    )

    // Merge, not replace: fields absent from the tick keep their prior values.
    expect($currentUsage.get()).toEqual({ ...BASELINE, context_percent: 42, input: 1200, total: 1280 })
    expect(sessionStates.get(SID)?.usage).toEqual({
      ...BASELINE,
      context_percent: 42,
      input: 1200,
      total: 1280
    })
  })

  it('caches a background session.usage tick without overwriting the primary status bar', async () => {
    await mountStream()

    act(() =>
      handleEvent!({
        payload: { usage: { input: 9999, total: 9999 } },
        session_id: 'background-session',
        type: 'session.usage'
      })
    )

    expect($currentUsage.get()).toEqual(BASELINE)
    expect(sessionStates.get('background-session')?.usage).toEqual({
      calls: 0,
      input: 9999,
      output: 0,
      total: 9999
    })
  })

  it('applies message.complete usage from the focused session', async () => {
    await mountStream()

    act(() =>
      handleEvent!({
        payload: { text: 'done', usage: { calls: 3, input: 1500, output: 90, total: 1590 } },
        session_id: SID,
        type: 'message.complete'
      })
    )

    expect($currentUsage.get()).toEqual({ calls: 3, input: 1500, output: 90, total: 1590 })
  })

  it('ignores message.complete usage from a background session', async () => {
    await mountStream()

    act(() =>
      handleEvent!({
        payload: { text: 'done', usage: { calls: 9, input: 9999, output: 999, total: 9999 } },
        session_id: 'background-session',
        type: 'message.complete'
      })
    )

    expect($currentUsage.get()).toEqual(BASELINE)
  })
})
