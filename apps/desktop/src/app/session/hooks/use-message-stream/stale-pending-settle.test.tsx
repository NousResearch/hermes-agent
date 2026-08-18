// A turn that ends WITHOUT its message.complete (turn crash, reconnect gap,
// steer race) used to leave its streaming bubble pending:true forever. The
// next user message then landed after it, stranding a live thinking indicator
// mid-transcript — the dither block anywhere but the tail. session.info
// running=false is the turn's finally-block signal and the only settle edge
// those paths still emit, so it must finalize the bubble.
import { QueryClient } from '@tanstack/react-query'
import { act, cleanup, render } from '@testing-library/react'
import { useEffect, useRef } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ClientSessionState } from '@/app/types'
import { createClientSessionState } from '@/lib/chat-runtime'
import type { RpcEvent } from '@/types/hermes'

import { STREAM_DELTA_FLUSH_MS } from './utils'

import { useMessageStream } from './index'

const SID = 'stale-pending-session'

let handleEvent: ((event: RpcEvent) => void) | null = null
let states: Map<string, ClientSessionState>

function Harness() {
  const activeSessionIdRef = useRef<string | null>(SID)
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
    states = sessionStateByRuntimeIdRef.current
  }, [stream.handleGatewayEvent])

  return null
}

async function mountHarness() {
  vi.useFakeTimers()
  render(<Harness />)
  await act(async () => {
    await Promise.resolve()
  })
}

const flushDeltas = async () => {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(STREAM_DELTA_FLUSH_MS)
  })
}

const emit = (event: RpcEvent) => act(() => handleEvent?.(event))

describe('turn end without message.complete (session.info running=false)', () => {
  beforeEach(() => {
    handleEvent = null
    states = new Map()
  })

  afterEach(() => {
    cleanup()
    vi.useRealTimers()
    vi.restoreAllMocks()
  })

  it('settles a streaming bubble that kept text', async () => {
    await mountHarness()

    emit({ session_id: SID, type: 'message.start', payload: {} })
    emit({ payload: { text: 'partial answer' }, session_id: SID, type: 'message.delta' })
    await flushDeltas()

    expect(states.get(SID)?.messages.at(-1)?.pending).toBe(true)

    emit({ payload: { running: false }, session_id: SID, type: 'session.info' })

    const state = states.get(SID)
    const tail = state?.messages.at(-1)

    expect(tail?.role).toBe('assistant')
    expect(tail?.pending).toBe(false)
    expect(tail?.parts).toMatchObject([{ type: 'text', text: 'partial answer' }])
    expect(state?.streamId).toBeNull()
    expect(state?.busy).toBe(false)
  })

  it('drops an empty streaming placeholder instead of stranding it', async () => {
    await mountHarness()

    emit({ session_id: SID, type: 'message.start', payload: {} })
    // A tool row seeds the bubble but no text ever arrives.
    emit({
      payload: { args: { command: 'true' }, name: 'terminal', tool_id: 't1' },
      session_id: SID,
      type: 'tool.start'
    })
    emit({
      payload: { name: 'terminal', result: 'ok', tool_id: 't1' },
      session_id: SID,
      type: 'tool.complete'
    })

    emit({ payload: { running: false }, session_id: SID, type: 'session.info' })

    const state = states.get(SID)

    // Same math as Stop: an empty-text placeholder is dropped, nothing stays
    // pending, and the stream binding is released.
    expect(state?.messages.every(message => !message.pending)).toBe(true)
    expect(state?.streamId).toBeNull()
  })
})
