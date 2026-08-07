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

describe('useMessageStream moa.reference accumulation (#64658)', () => {
  beforeEach(() => {
    handleEvent = null
    latestState = null
  })

  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
  })

  it('keeps every reference model labelled block instead of only the latest one', async () => {
    await mountStream()

    emit('moa.reference', { count: 2, index: 1, label: 'model-a', text: 'advice-a' })
    emit('moa.reference', { count: 2, index: 2, label: 'model-b', text: 'advice-b' })

    const text = reasoningText()

    expect(text).toContain('model-a')
    expect(text).toContain('advice-a')
    expect(text).toContain('model-b')
    expect(text).toContain('advice-b')
  })

  it('handles a single-reference MoA turn (count=1) without regression', async () => {
    await mountStream()

    emit('moa.reference', { count: 1, index: 1, label: 'model-a', text: 'only-advice' })

    const text = reasoningText()

    expect(text).toContain('model-a')
    expect(text).toContain('only-advice')
  })

  it('accumulates three or more references in order', async () => {
    await mountStream()

    emit('moa.reference', { count: 3, index: 1, label: 'model-a', text: 'advice-a' })
    emit('moa.reference', { count: 3, index: 2, label: 'model-b', text: 'advice-b' })
    emit('moa.reference', { count: 3, index: 3, label: 'model-c', text: 'advice-c' })

    const text = reasoningText()

    const orderOk =
      text.indexOf('advice-a') < text.indexOf('advice-b') && text.indexOf('advice-b') < text.indexOf('advice-c')

    expect(text).toContain('advice-a')
    expect(text).toContain('advice-b')
    expect(text).toContain('advice-c')
    expect(orderOk).toBe(true)
  })
})
