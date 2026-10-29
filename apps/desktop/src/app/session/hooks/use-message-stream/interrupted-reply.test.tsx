import type { GatewayEvent } from '@hermes/shared'
// #121594: Stop seals the live bubble at the click and drops every later
// message.delta, but the agent keeps streaming until it honours the interrupt
// and persists everything it delivered (state.db and the next turn's context).
// Its interrupted message.complete carries that persisted partial. Contract:
// the screen shows exactly what the session saved — extend-only, never
// shortened or rewritten.
import { act, cleanup } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { finalizeUserInterruptedMessages } from '@/app/session/hooks/use-prompt-actions/rewind'
import { chatMessageText } from '@/lib/chat-messages'

import { type MessageStreamHarness, renderMessageStream } from './test-harness'
import { STREAM_DELTA_FLUSH_MS } from './utils'

const SID = 'interrupted-reply-session'

let stream: MessageStreamHarness

async function mountHarness() {
  vi.useFakeTimers()
  stream = renderMessageStream(SID)
  await act(async () => {
    await Promise.resolve()
  })
}

const flushDeltas = async () => {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(STREAM_DELTA_FLUSH_MS)
  })
}

const emit = (event: GatewayEvent) => act(() => stream.handleEvent(event))

// The state transform both Stop paths (use-prompt-actions and the session
// tile) apply at the click.
const pressStop = () =>
  act(() => {
    const state = stream.state()
    stream.states.set(SID, {
      ...state,
      messages: finalizeUserInterruptedMessages(state.messages, state.streamId),
      busy: false,
      awaitingResponse: false,
      streamId: null,
      pendingBranchGroup: null,
      needsInput: false,
      interrupted: true,
      turnStartedAt: null,
      turnLive: false
    })
  })

const assistantTexts = () =>
  stream
    .state()
    .messages.filter(message => message.role === 'assistant' && !message.hidden)
    .map(message => chatMessageText(message))

describe('Stop: the screen shows the partial the session saves (#121594)', () => {
  afterEach(() => {
    cleanup()
    vi.useRealTimers()
    vi.restoreAllMocks()
  })

  it('late deltas after Stop reach the bubble through the interrupted completion', async () => {
    await mountHarness()
    emit({ payload: {}, session_id: SID, type: 'message.start' })
    emit({ payload: { text: 'The quick brown ' }, session_id: SID, type: 'message.delta' })
    await flushDeltas()

    pressStop()
    // In flight when Stop landed: dropped by the renderer, persisted by the agent.
    emit({ payload: { text: 'fox jumps ' }, session_id: SID, type: 'message.delta' })
    await flushDeltas()
    expect(assistantTexts()).toEqual(['The quick brown '])

    emit({
      payload: { status: 'interrupted', text: 'The quick brown fox jumps' },
      session_id: SID,
      type: 'message.complete'
    })

    expect(assistantTexts()).toEqual(['The quick brown fox jumps'])
    expect(stream.state().busy).toBe(false)
  })

  it('adds the bubble when Stop landed before anything was painted', async () => {
    await mountHarness()
    emit({ payload: {}, session_id: SID, type: 'message.start' })
    // Queued for the next flush, not yet painted.
    emit({ payload: { text: 'Hello' }, session_id: SID, type: 'message.delta' })
    pressStop()
    await flushDeltas()
    expect(assistantTexts()).toEqual([])

    emit({ payload: { status: 'interrupted', text: 'Hello there' }, session_id: SID, type: 'message.complete' })

    expect(assistantTexts()).toEqual(['Hello there'])
  })

  it('never shortens or rewrites the shown partial', async () => {
    await mountHarness()
    emit({ payload: {}, session_id: SID, type: 'message.start' })
    emit({ payload: { text: 'Alpha beta gamma' }, session_id: SID, type: 'message.delta' })
    await flushDeltas()
    pressStop()

    emit({ payload: { status: 'interrupted', text: 'Alpha beta' }, session_id: SID, type: 'message.complete' })
    expect(assistantTexts()).toEqual(['Alpha beta gamma'])

    emit({ payload: { status: 'interrupted', text: 'Something else' }, session_id: SID, type: 'message.complete' })
    expect(assistantTexts()).toEqual(['Alpha beta gamma'])
  })

  it('does not repeat the pre-tool text when Stop lands during a tool', async () => {
    await mountHarness()
    emit({ payload: {}, session_id: SID, type: 'message.start' })
    emit({ payload: { text: 'Let me check.' }, session_id: SID, type: 'message.delta' })
    await flushDeltas()
    emit({
      payload: { args: { command: 'ls' }, name: 'terminal', tool_id: 'call-1' },
      session_id: SID,
      type: 'tool.start'
    })
    pressStop()

    emit({ payload: { status: 'interrupted', text: 'Let me check.' }, session_id: SID, type: 'message.complete' })

    expect(assistantTexts()).toEqual(['Let me check.'])
  })
})
