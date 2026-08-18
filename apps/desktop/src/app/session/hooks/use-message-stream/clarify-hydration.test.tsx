import { QueryClient } from '@tanstack/react-query'
import { act, cleanup, render, waitFor } from '@testing-library/react'
import { type MutableRefObject, useEffect, useRef } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ClientSessionState } from '@/app/types'
import { createClientSessionState } from '@/lib/chat-runtime'
import { $clarifyRequests, clearClarifyRequest } from '@/store/clarify'
import { onScrollToBottomRequest } from '@/store/thread-scroll'
import type { RpcEvent } from '@/types/hermes'

import { useMessageStream } from './index'

// A `clarify.request` must leave an answerable inline row even when the
// `tool.start` that normally mounts it was missed (stream reconnect /
// hydration race). Without it the sidebar says "needs input" but the
// transcript has nowhere to render the choices, so the agent blocks forever.

const SID = 'session-1'

let handleEvent: ((event: RpcEvent) => void) | null = null
let stateRef: MutableRefObject<Map<string, ClientSessionState>> | null = null
let stopScrollListener: (() => void) | null = null

const scrollToBottom = vi.fn()

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
    stateRef = sessionStateByRuntimeIdRef
  }, [stream.handleGatewayEvent])

  return null
}

async function mountStream() {
  render(<Harness />)
  await waitFor(() => expect(handleEvent).not.toBeNull())
}

const clarifyRequest = (payload: Record<string, unknown>) =>
  act(() => handleEvent!({ payload, session_id: SID, type: 'clarify.request' }))

const toolStart = (payload: Record<string, unknown>) =>
  act(() => handleEvent!({ payload, session_id: SID, type: 'tool.start' }))

function clarifyParts() {
  const messages = stateRef?.current.get(SID)?.messages ?? []

  return messages.flatMap(m => m.parts).filter(p => p.type === 'tool-call' && p.toolName === 'clarify')
}

describe('clarify.request stream hydration', () => {
  beforeEach(() => {
    handleEvent = null
    stateRef = null
    clearClarifyRequest()
    scrollToBottom.mockClear()
    stopScrollListener = onScrollToBottomRequest(scrollToBottom)
  })

  afterEach(() => {
    cleanup()
    clearClarifyRequest()
    stopScrollListener?.()
    stopScrollListener = null
    vi.restoreAllMocks()
  })

  it('mounts an answerable clarify row when the tool.start row was missed', async () => {
    await mountStream()

    clarifyRequest({ choices: ['yes', 'no'], question: 'Ship it?', request_id: 'req-1' })

    const parts = clarifyParts()
    expect(parts).toHaveLength(1)
    expect(parts[0].type === 'tool-call' && parts[0].toolCallId).toBe('req-1')
    expect(parts[0].type === 'tool-call' && parts[0].args).toMatchObject({
      choices: ['yes', 'no'],
      question: 'Ship it?'
    })
  })

  it('reveals a clarify prompt raised by the active session', async () => {
    await mountStream()

    clarifyRequest({ choices: ['yes', 'no'], question: 'Ship it?', request_id: 'req-reveal' })

    expect(scrollToBottom).toHaveBeenCalledOnce()
  })

  it('does not move the active thread for a background session clarify', async () => {
    await mountStream()

    act(() =>
      handleEvent!({
        payload: { choices: ['yes', 'no'], question: 'Ship it?', request_id: 'req-background' },
        session_id: 'session-background',
        type: 'clarify.request'
      })
    )

    expect(scrollToBottom).not.toHaveBeenCalled()
  })

  it('preserves multi-select through the store and hydrated tool row', async () => {
    await mountStream()

    clarifyRequest({
      choices: ['read', 'write'],
      multi_select: true,
      question: 'Which permissions?',
      request_id: 'req-multi'
    })

    expect($clarifyRequests.get()[SID]?.multiSelect).toBe(true)

    const part = clarifyParts()[0]
    expect(part?.type).toBe('tool-call')

    if (part?.type !== 'tool-call') {
      throw new Error('Expected a hydrated clarify tool call')
    }

    expect(part.args).toMatchObject({
      choices: ['read', 'write'],
      multi_select: true,
      question: 'Which permissions?'
    })
  })

  it('merges with the real tool.start row even though its id differs from the request id', async () => {
    await mountStream()

    // Reality: tool.start carries the model's tool_call_id, clarify.request a
    // separately-generated request_id. They must still collapse to ONE card
    // (correlated by question), not two.
    toolStart({ args: { choices: ['a'], question: 'Pick' }, name: 'clarify', tool_id: 'call-abc' })
    clarifyRequest({ choices: ['a'], question: 'Pick', request_id: 'req-2' })

    expect(clarifyParts()).toHaveLength(1)
  })

  it('does not duplicate when clarify.request arrives before the tool.start row', async () => {
    await mountStream()

    clarifyRequest({ choices: ['a'], question: 'Pick', request_id: 'req-3' })
    toolStart({ args: { choices: ['a'], question: 'Pick' }, name: 'clarify', tool_id: 'call-xyz' })

    expect(clarifyParts()).toHaveLength(1)
  })

  it('merges a BATCH tool.start row with its clarify.request (no top-level question)', async () => {
    await mountStream()

    // The batch shape: tool args carry `questions`, no top-level `question`.
    // The correlation key must come from the question list, or the two ids
    // mount two cards (the duplicate seen in the field).
    toolStart({
      args: { questions: [{ question: 'Drink?' }, { question: 'Productive when?' }] },
      name: 'clarify',
      tool_id: 'call-batch'
    })
    clarifyRequest({
      questions: [
        { qid: 'q0', question: 'Drink?' },
        { qid: 'q1', question: 'Productive when?' }
      ],
      request_id: 'req-batch'
    })

    expect(clarifyParts()).toHaveLength(1)
  })

  it('does not duplicate when the batch clarify.request arrives before tool.start', async () => {
    await mountStream()

    clarifyRequest({
      questions: [
        { qid: 'q0', question: 'Drink?' },
        { qid: 'q1', question: 'Productive when?' }
      ],
      request_id: 'req-batch-2'
    })
    toolStart({
      args: { questions: [{ question: 'Drink?' }, { question: 'Productive when?' }] },
      name: 'clarify',
      tool_id: 'call-batch-2'
    })

    expect(clarifyParts()).toHaveLength(1)
  })
})
