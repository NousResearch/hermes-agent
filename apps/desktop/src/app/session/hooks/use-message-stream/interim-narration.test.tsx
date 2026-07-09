import { QueryClient } from '@tanstack/react-query'
import { act, cleanup, render, waitFor } from '@testing-library/react'
import { useEffect, useRef } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ClientSessionState } from '@/app/types'
import { createClientSessionState } from '@/lib/chat-runtime'
import { $keepInterimAssistantMessages, setKeepInterimAssistantMessages } from '@/store/session'
import type { RpcEvent } from '@/types/hermes'

import { useMessageStream } from './index'

// Integration test for the interim-narration fix: drive the REAL hook through a
// multi-step turn (delta → tool → delta → complete) via handleGatewayEvent and
// assert the mid-turn narration survives on the finalized message parts. This
// exercises the wiring (the $keepInterimAssistantMessages atom + its read in
// completeAssistantMessage) that the pure mergeFinalAssistantText unit tests in
// lib/chat-messages.test.ts cannot cover.

const SID = 'session-1'

let handleEvent: ((event: RpcEvent) => void) | null = null
let readState: (() => ClientSessionState | undefined) | null = null

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
    readState = () => sessionStateByRuntimeIdRef.current.get(SID)
  }, [stream.handleGatewayEvent])

  return null
}

async function mountStream() {
  render(<Harness />)
  await waitFor(() => expect(handleEvent).not.toBeNull())
}

function fire(event: RpcEvent) {
  act(() => handleEvent!(event))
}

// Deltas are queued but flushed synchronously at tool.start and message.complete
// boundaries (flushQueuedDeltas is called in those handlers), so a multi-step
// turn resolves deterministically without advancing any timer.
function runMultiStepTurn() {
  fire({ payload: {}, session_id: SID, type: 'message.start' })
  fire({ payload: { text: 'Let me check the repo.' }, session_id: SID, type: 'message.delta' })
  fire({ payload: { name: 'terminal', tool_id: 'tc-1' }, session_id: SID, type: 'tool.start' })
  fire({
    payload: { name: 'terminal', tool_id: 'tc-1', result: { output: 'no ci', exit_code: 0 } },
    session_id: SID,
    type: 'tool.complete'
  })
  fire({ payload: { text: 'No CI here.' }, session_id: SID, type: 'message.delta' })
  fire({ payload: { text: 'No CI here.' }, session_id: SID, type: 'message.complete' })
}

function textPartTexts(state: ClientSessionState | undefined): string[] {
  const message = state?.messages.at(-1)

  return (message?.parts ?? [])
    .filter((p): p is { type: 'text'; text: string } => p.type === 'text')
    .map(p => p.text)
}

function partTypes(state: ClientSessionState | undefined): string[] {
  return (state?.messages.at(-1)?.parts ?? []).map(p => p.type)
}

describe('useMessageStream interim narration on completion', () => {
  const initialKeepInterim = $keepInterimAssistantMessages.get()

  beforeEach(() => {
    handleEvent = null
    readState = null
  })

  afterEach(() => {
    cleanup()
    setKeepInterimAssistantMessages(initialKeepInterim)
    vi.restoreAllMocks()
  })

  it('keeps mid-turn narration when interim_assistant_messages is on (default)', async () => {
    setKeepInterimAssistantMessages(true)
    await mountStream()

    runMultiStepTurn()

    expect(partTypes(readState!())).toEqual(['text', 'tool-call', 'text'])
    expect(textPartTexts(readState!())).toEqual(['Let me check the repo.', 'No CI here.'])
  })

  it('collapses to the final message when interim_assistant_messages is off', async () => {
    setKeepInterimAssistantMessages(false)
    await mountStream()

    runMultiStepTurn()

    // Lean mode: the interim "Let me check the repo." narration is dropped; only
    // the tool part and the final text survive.
    expect(partTypes(readState!())).toEqual(['tool-call', 'text'])
    expect(textPartTexts(readState!())).toEqual(['No CI here.'])
  })
})
