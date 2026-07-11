import { QueryClient } from '@tanstack/react-query'
import { act, cleanup, render, waitFor } from '@testing-library/react'
import { useEffect, useRef } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ClientSessionState } from '@/app/types'
import { COMMENTARY_PART_TYPE, commentaryPartText } from '@/lib/chat-messages'
import { createClientSessionState } from '@/lib/chat-runtime'
import type { RpcEvent } from '@/types/hermes'

import { useMessageStream } from './index'

const SID = 'session-1'

let handleEvent: ((event: RpcEvent) => void) | null = null
let sessionStates: Map<string, ClientSessionState> | null = null

function Harness() {
  const activeSessionIdRef = useRef<string | null>(SID)
  const sessionStateByRuntimeIdRef = useRef(new Map<string, ClientSessionState>())
  const queryClientRef = useRef(new QueryClient())

  sessionStates = sessionStateByRuntimeIdRef.current

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

const send = (event: RpcEvent) => act(() => handleEvent!(event))

const assistantMessage = () => {
  const messages = sessionStates?.get(SID)?.messages ?? []

  return messages.find(message => message.role === 'assistant')
}

describe('useMessageStream commentary lane', () => {
  beforeEach(() => {
    handleEvent = null
    sessionStates = null
  })

  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
  })

  it('routes commentary.delta into a commentary part, never the reasoning part', async () => {
    await mountStream()

    send({ payload: { text: 'Planning the fix' }, session_id: SID, type: 'reasoning.delta' })
    send({ payload: { text: 'Reading the screenshot first.' }, session_id: SID, type: 'commentary.delta' })
    send({ payload: { text: 'The total is $42.' }, session_id: SID, type: 'message.complete' })

    const message = assistantMessage()
    expect(message).toBeDefined()

    const reasoningParts = message!.parts.filter(part => part.type === 'reasoning')
    const commentaryParts = message!.parts.filter(part => part.type === COMMENTARY_PART_TYPE)
    const textParts = message!.parts.filter(part => part.type === 'text')

    expect(reasoningParts).toHaveLength(1)
    expect((reasoningParts[0] as { text: string }).text).toBe('Planning the fix')
    expect(commentaryParts).toHaveLength(1)
    expect(commentaryPartText(commentaryParts[0])).toBe('Reading the screenshot first.')
    expect(textParts).toHaveLength(1)
    expect((textParts[0] as { text: string }).text).toBe('The total is $42.')
  })

  it('preserves reasoning → commentary → tool → commentary → answer ordering', async () => {
    await mountStream()

    send({ payload: { text: 'Planning' }, session_id: SID, type: 'reasoning.delta' })
    send({ payload: { text: 'Checking the logs first.' }, session_id: SID, type: 'commentary.delta' })
    // Tool events flush queued deltas synchronously, freezing the segment.
    send({
      payload: { name: 'terminal', tool_id: 'call_1' },
      session_id: SID,
      type: 'tool.start'
    })
    send({
      payload: { name: 'terminal', result: 'ok', tool_id: 'call_1' },
      session_id: SID,
      type: 'tool.complete'
    })
    send({ payload: { text: 'Now writing the summary.' }, session_id: SID, type: 'commentary.delta' })
    send({ payload: { text: 'All done.' }, session_id: SID, type: 'message.complete' })

    const message = assistantMessage()
    expect(message).toBeDefined()

    const partTypes = message!.parts.map(part => part.type)
    expect(partTypes).toEqual(['reasoning', COMMENTARY_PART_TYPE, 'tool-call', COMMENTARY_PART_TYPE, 'text'])
    expect(commentaryPartText(message!.parts[1])).toBe('Checking the logs first.')
    expect(commentaryPartText(message!.parts[3])).toBe('Now writing the summary.')
    expect((message!.parts[4] as { text: string }).text).toBe('All done.')
  })

  it('keeps commentary above the answer when both lanes land in one flush window', async () => {
    await mountStream()

    // No tool event between these, so both lanes batch into the same
    // delta-flush window; commentary must still render above the answer.
    send({ payload: { text: 'Wrapping up now.' }, session_id: SID, type: 'commentary.delta' })
    send({ payload: { text: 'The answer.' }, session_id: SID, type: 'message.delta' })
    send({ payload: { text: 'The answer.' }, session_id: SID, type: 'message.complete' })

    const message = assistantMessage()
    expect(message!.parts.map(part => part.type)).toEqual([COMMENTARY_PART_TYPE, 'text'])
    expect(commentaryPartText(message!.parts[0])).toBe('Wrapping up now.')
  })

  it('coalesces consecutive commentary deltas into one part per segment', async () => {
    await mountStream()

    send({ payload: { text: 'Reading ' }, session_id: SID, type: 'commentary.delta' })
    send({ payload: { text: 'the file.' }, session_id: SID, type: 'commentary.delta' })
    send({ payload: { text: 'Done.' }, session_id: SID, type: 'message.complete' })

    const message = assistantMessage()
    const commentaryParts = message!.parts.filter(part => part.type === COMMENTARY_PART_TYPE)

    expect(commentaryParts).toHaveLength(1)
    expect(commentaryPartText(commentaryParts[0])).toBe('Reading the file.')
  })
})
