import { act, cleanup } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { textPart } from '@/lib/chat-messages'
import { createClientSessionState } from '@/lib/chat-runtime'
import { $notifications, clearNotifications } from '@/store/notifications'

import { type MessageStreamHarness, renderMessageStream } from './test-harness'

const SID = 'rt-new-session'

let stream: MessageStreamHarness

function mountStream() {
  stream = renderMessageStream(SID)
}

/** Seed the session as it looks right after a first-message submit: the
 *  optimistic user row is present, the turn is awaiting its response. */
function seedOptimisticFirstMessage() {
  stream.states.set(SID, {
    ...createClientSessionState('stored-new-session', [
      { id: 'user-123-abc', role: 'user', parts: [textPart('first message of a new chat')] }
    ]),
    busy: true,
    awaitingResponse: true
  })
}

describe('useMessageStream agent-init error surfacing (#63078)', () => {
  beforeEach(() => {
    clearNotifications()
  })

  afterEach(() => {
    cleanup()
    clearNotifications()
    vi.restoreAllMocks()
  })

  it('renders an agent-init failure as a visible in-transcript error and keeps the optimistic first message', async () => {
    mountStream()
    seedOptimisticFirstMessage()

    act(() =>
      stream.handleEvent({
        payload: {
          message:
            'agent initialization timed out after 601s — your message was not sent; retry once the session is ready'
        },
        session_id: SID,
        type: 'error'
      })
    )

    const state = stream.state()

    // The user's optimistic first message must survive — the failure mode of
    // #63078 was the message silently vanishing into a blank session.
    const userRows = state.messages.filter(m => m.role === 'user')
    expect(userRows).toHaveLength(1)
    expect(userRows[0]!.id).toBe('user-123-abc')

    // The failure is VISIBLE in the session view: an assistant error bubble...
    const errorRows = state.messages.filter(m => m.role === 'assistant' && m.error)
    expect(errorRows).toHaveLength(1)
    expect(errorRows[0]!.error).toContain('your message was not sent')

    // ...and the composer is released (no forever-spinner on a dead turn).
    expect(state.busy).toBe(false)
    expect(state.awaitingResponse).toBe(false)

    // A global toast also fired (turn-ending errors are easy to miss inline).
    // The server already sends plain, actionable copy for an agent-init
    // failure, so it IS the toast message — not demoted to a detail line
    // under a generic gloss.
    const toast = $notifications.get().find(n => n.kind === 'error' && n.message.includes('was not sent'))
    expect(toast).toBeDefined()
    expect(toast!.detail).toBeUndefined()
  })
  it('merges the terminal error frame into the card the agent-init error event already painted', async () => {
    mountStream()
    seedOptimisticFirstMessage()

    act(() =>
      stream.handleEvent({
        payload: { message: 'Hermes could not start the assistant for this session. Details: no provider.' },
        session_id: SID,
        type: 'error'
      })
    )

    act(() =>
      stream.handleEvent({
        payload: {
          error: 'no provider',
          error_surface: { code: 'agent_init_failed', layer: 'runtime', retryable: true },
          status: 'error',
          text: 'no provider'
        },
        session_id: SID,
        type: 'message.complete'
      })
    )

    const errorRows = stream.state().messages.filter(m => m.role === 'assistant' && m.error)
    expect(errorRows).toHaveLength(1)
    expect(errorRows[0]!.errorSurface?.code).toBe('agent_init_failed')
  })
  it('keeps one card when the terminal error frame arrives before the agent-init error event', async () => {
    mountStream()
    seedOptimisticFirstMessage()

    act(() =>
      stream.handleEvent({
        payload: {
          error: 'no provider',
          error_surface: { code: 'agent_init_failed', layer: 'runtime', retryable: true },
          status: 'error',
          text: 'no provider'
        },
        session_id: SID,
        type: 'message.complete'
      })
    )

    act(() =>
      stream.handleEvent({
        payload: { message: 'Hermes could not start the assistant for this session. Details: no provider.' },
        session_id: SID,
        type: 'error'
      })
    )

    const errorRows = stream.state().messages.filter(m => m.role === 'assistant' && m.error)
    expect(errorRows).toHaveLength(1)
    expect(errorRows[0]!.error).toContain('could not start the assistant')
    expect(errorRows[0]!.errorSurface?.code).toBe('agent_init_failed')
  })
})
