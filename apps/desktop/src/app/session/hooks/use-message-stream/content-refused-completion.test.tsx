import { act, cleanup } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { ClientSessionState } from '@/app/types'
import { type ChatMessage, chatMessageText, textPart } from '@/lib/chat-messages'
import { createClientSessionState, toRuntimeMessage } from '@/lib/chat-runtime'

import { type MessageStreamHarness, renderMessageStream } from './test-harness'

const SID = 'session-1'
const PROMPT = 'the request the filter blocked'
const STREAM_ID = `assistant-stream-${SID}`

let stream: MessageStreamHarness

// The turn as the UI holds it the instant a content-policy refusal lands live:
// the user's optimistic prompt bubble and the still-pending assistant stream.
function seededStates(): Map<string, ClientSessionState> {
  const userRow: ChatMessage = { id: 'user-123', role: 'user', parts: [textPart(PROMPT)], timestamp: 1 }
  const pending: ChatMessage = { id: STREAM_ID, role: 'assistant', parts: [textPart('')], pending: true, timestamp: 2 }

  const state: ClientSessionState = {
    ...createClientSessionState(SID, [userRow, pending]),
    streamId: STREAM_ID,
    sawAssistantPayload: true,
    awaitingResponse: true,
    busy: true
  }

  return new Map([[SID, state]])
}

function mountStream() {
  stream = renderMessageStream(SID, { states: seededStates() })
}

// The gateway's terminal message.complete frame for a safety-filter refusal
// (gateway/session_settlement_recovery.py completion_payload): status "error",
// `failure_reason` stamped content_policy_blocked, the same code on the surface.
const refuse = () =>
  act(() =>
    stream.handleEvent({
      payload: {
        status: 'error',
        error: "the AI model service's safety filter rejected the request",
        failure_reason: 'content_policy_blocked',
        error_surface: { layer: 'provider', code: 'content_policy_blocked', retryable: false }
      },
      session_id: SID,
      type: 'message.complete'
    })
  )

function messages() {
  return stream.state(SID).messages
}

describe('live content-policy refusal completion', () => {
  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
  })

  it('replaces the live turn with the greyed content-refused row, not an error bubble beside the user prompt', async () => {
    mountStream()
    await refuse()

    const rows = messages()

    // A single refused row stands in for the whole turn (no normal user bubble,
    // no assistant error card) — the exact shape stored hydration produces.
    const refused = rows.filter(message => message.contentRefused)
    expect(refused).toHaveLength(1)
    expect(refused[0].role).toBe('system')
    expect(chatMessageText(refused[0])).toBe(PROMPT)
    expect(refused[0].contentRefused).toEqual({})

    // The blue user bubble is gone (its text now lives inside the greyed block),
    // and no assistant error card settled beside it.
    expect(rows.some(message => message.role === 'user')).toBe(false)
    expect(rows.some(message => message.role === 'assistant' && message.error)).toBe(false)

    // Turn is settled — the chat is not locked.
    expect(stream.state(SID).busy).toBe(false)
    expect(stream.state(SID).awaitingResponse).toBe(false)
  })

  it('carries the refusal into the render contract the SystemMessage consumes', async () => {
    mountStream()
    await refuse()

    const refused = messages().find(message => message.contentRefused)
    const runtime = toRuntimeMessage(refused!)

    // SystemMessage reads role + metadata.custom.contentRefused + the text part.
    expect(runtime.role).toBe('system')
    expect((runtime.metadata?.custom as { contentRefused?: unknown }).contentRefused).toEqual({})
    const [part] = runtime.content as { type: string; text: string }[]
    expect(part).toEqual({ type: 'text', text: PROMPT })
  })
})
