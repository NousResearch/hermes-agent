// Regression for the scoped settle-path pending clear (PR #92730 round 2).
//
// Round 1's blanket wipe — nextMessages.map(m => m.role === 'user' && m.pending
// ? { ...m, pending: false } : m) — ran over EVERY user row in the session.
// A queued turn B's optimistic user row can sit below turn A's streaming
// assistant bubble when A settles; the blanket pass stripped B's protection
// before B persisted/reconciled. The fix retires only pending user rows ABOVE
// the settled assistant row (index boundary); locating no tracked assistant
// row clears nothing (deferred retirement via finalizeInterruptedMessages).
//
// These specs drive the REAL reducer through handleGatewayEvent — no state
// hand-mapping — so the assertions bite on the actual settle logic.
import { act, cleanup } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { ClientSessionState } from '@/app/types'
import { createClientSessionState } from '@/lib/chat-runtime'
import type { RpcEvent } from '@/types/hermes'

import { type MessageStreamHarness, renderMessageStream } from './test-harness'

const SID = 'scoped-pending-settle-session'

let stream: MessageStreamHarness
let states: Map<string, ClientSessionState>

function seedQueuedFixture(): void {
  const state = createClientSessionState()
  state.messages = [
    { id: 'user-current', role: 'user', parts: [{ type: 'text', text: 'current' }], pending: true },
    { id: 'assistant-stream-current', role: 'assistant', parts: [{ type: 'text', text: '' }], pending: true },
    { id: 'user-next', role: 'user', parts: [{ type: 'text', text: 'queued next' }], pending: true }
  ]
  state.streamId = 'assistant-stream-current'
  states.set(SID, state)
}

async function mountHarness() {
  vi.useFakeTimers()
  stream = renderMessageStream(SID)
  states = stream.states
  seedQueuedFixture()
  await act(async () => {
    await Promise.resolve()
  })
}

const emit = (event: RpcEvent) => act(() => stream.handleEvent(event))

describe('settle-path pending clearing is scoped to the settling turn', () => {
  afterEach(() => {
    cleanup()
    vi.useRealTimers()
    vi.restoreAllMocks()
  })

  it('retires the current turn user row but leaves a later queued user row protected on completion', async () => {
    await mountHarness()

    emit({ payload: { text: 'answer' }, session_id: SID, type: 'message.complete' })

    const after = states.get(SID)!
    const userCurrent = after.messages.find(m => m.id === 'user-current')
    const userNext = after.messages.find(m => m.id === 'user-next')

    // Turn A settled: its optimistic prompt is retired...
    expect(userCurrent?.pending).toBe(false)
    // ...and turn B's queued prompt keeps its cache protection until it persists/reconciles.
    expect(userNext?.pending).toBe(true)
    expect(after.busy).toBe(false)
  })

  it('keeps the queued user row protected when the turn completes with an error frame', async () => {
    await mountHarness()

    emit({
      payload: { status: 'error', error: 'connection reset mid-stream' },
      session_id: SID,
      type: 'message.complete'
    })

    const after = states.get(SID)!
    const userCurrent = after.messages.find(m => m.id === 'user-current')
    const userNext = after.messages.find(m => m.id === 'user-next')

    // Turn A failed but had a tracked stream row: rows above the failed
    // assistant row are retired...
    expect(userCurrent?.pending).toBe(false)
    // ...and the queued row below the boundary keeps its protection.
    expect(userNext?.pending).toBe(true)
  })

  it('clears nothing when a completion arrives with no tracked stream row (deferred retirement)', async () => {
    // No message.start ever seeded a tracked bubble for this turn: the
    // completion falls into the append-at-tail branch. The settle marks no
    // pre-existing turn in `prev`, so pending user rows stay protected.
    stream = renderMessageStream(SID)
    states = stream.states
    const state = createClientSessionState()
    state.messages = [
      { id: 'user-a', role: 'user', parts: [{ type: 'text', text: 'first' }], pending: true },
      { id: 'user-b', role: 'user', parts: [{ type: 'text', text: 'second' }], pending: true }
    ]
    state.streamId = null
    states.set(SID, state)

    emit({ payload: { text: 'answer' }, session_id: SID, type: 'message.complete' })

    const after = states.get(SID)!
    const users = after.messages.filter(m => m.role === 'user')
    expect(users).toHaveLength(2)
    expect(users.every(m => m.pending === true)).toBe(true)
    // The reply still appended normally (a freshly appended settled row
    // carries no pending flag at all).
    expect(after.messages.at(-1)?.role).toBe('assistant')
    expect(after.messages.at(-1)?.pending).not.toBe(true)
  })

  it('clears nothing on failure when the stream row was never tracked (deferred retirement)', () => {
    stream = renderMessageStream(SID)
    states = stream.states
    const state = createClientSessionState()
    state.messages = [
      { id: 'user-orphan', role: 'user', parts: [{ type: 'text', text: 'unacknowledged' }], pending: true },
      { id: 'user-next', role: 'user', parts: [{ type: 'text', text: 'queued next' }], pending: true }
      // Deliberately NO assistant row matching the fabricated assistant-error-* id.
    ]
    state.streamId = null
    states.set(SID, state)

    act(() => {
      // Top-level gateway 'error' events route to failAssistantMessage.
      stream.handleEvent({ payload: { message: 'boom' }, session_id: SID, type: 'error' })
    })

    const after = states.get(SID)!
    // Both pending rows survive — retirement is deferred to
    // finalizeInterruptedMessages on the next submit/interrupt.
    expect(after.messages.filter(m => m.role === 'user').every(m => m.pending === true)).toBe(true)
    // The failure still surfaced as its own errored assistant row.
    const errorRow = after.messages.find(m => m.role === 'assistant' && m.error)
    expect(errorRow).toBeDefined()
    expect(errorRow?.id).toContain('assistant-error-')
  })
})
