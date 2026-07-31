import { QueryClient } from '@tanstack/react-query'
import { act, cleanup, render, waitFor } from '@testing-library/react'
import { useEffect, useRef } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ClientSessionState } from '@/app/types'
import { createClientSessionState } from '@/lib/chat-runtime'
import type { RpcEvent } from '@/types/hermes'

import { useMessageStream } from './index'

const SID = 'session-1'
let handleEvent: ((event: RpcEvent) => void) | null = null
let latestState: ClientSessionState | null = null

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
      latestState = next

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

function reasoningText(): string {
  const message = latestState?.messages.at(-1)
  const part = message?.parts.find(p => p.type === 'reasoning')

  return part?.type === 'reasoning' ? part.text : ''
}

beforeEach(() => {
  handleEvent = null
  latestState = null
})

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

describe('useMessageStream moa.progress / moa.phase surfacing', () => {
  it('shows refs k/n lines in the reasoning block as references complete', async () => {
    await mountStream()

    emit('message.start')
    emit('moa.progress', { label: 'model-a', refs_done: 1, refs_total: 3 })
    emit('moa.progress', { label: 'model-b', refs_done: 2, refs_total: 3 })

    const text = reasoningText()
    expect(text).toContain('MoA refs 1/3 — model-a')
    expect(text).toContain('MoA refs 2/3 — model-b')
  })

  it('restarts the progress block on the first ref of a new fan-out', async () => {
    await mountStream()

    emit('message.start')
    emit('moa.progress', { label: 'stale', refs_done: 1, refs_total: 2 })
    emit('moa.progress', { label: 'stale-2', refs_done: 2, refs_total: 2 })
    // A later turn's fan-out starts over at 1/N — old lines must not linger.
    emit('moa.progress', { label: 'fresh', refs_done: 1, refs_total: 2 })

    const text = reasoningText()
    expect(text).toContain('MoA refs 1/2 — fresh')
    expect(text).not.toContain('stale')
  })

  it('appends the aggregating marker on moa.phase and ignores unknown phases', async () => {
    await mountStream()

    emit('message.start')
    emit('moa.progress', { label: 'model-a', refs_done: 1, refs_total: 1 })
    emit('moa.phase', { phase: 'reference', refs_done: 1, refs_total: 1 })
    expect(reasoningText()).not.toContain('aggregating')

    emit('moa.phase', { aggregator: 'agg-model', phase: 'aggregator', refs_done: 1, refs_total: 1 })
    expect(reasoningText()).toContain('MoA aggregating…')
  })

  it('a following moa.reference replaces the progress trail (self-cleaning)', async () => {
    await mountStream()

    emit('message.start')
    emit('moa.progress', { label: 'model-a', refs_done: 1, refs_total: 1 })
    emit('moa.phase', { phase: 'aggregator', refs_done: 1, refs_total: 1 })
    emit('moa.reference', { count: 1, index: 1, label: 'model-a', text: 'advice-a' })

    const text = reasoningText()
    expect(text).toContain('Reference 1/1 — model-a')
    expect(text).not.toContain('MoA refs')
    expect(text).not.toContain('aggregating')
  })
})
