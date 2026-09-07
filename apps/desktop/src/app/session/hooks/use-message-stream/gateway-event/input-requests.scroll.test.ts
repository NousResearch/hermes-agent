import { afterEach, describe, expect, it, vi } from 'vitest'

import { createClientSessionState } from '@/lib/chat-runtime'
import { clearClarifyRequest, sessionClarifyRequest } from '@/store/clarify'
import { onScrollToBottomRequest, resetThreadScroll, setThreadAtBottom } from '@/store/thread-scroll'

import { handleInputRequestEvent } from './input-requests'
import type { GatewayEventContext } from './types'

// A clarify.request for the ACTIVE session snaps the transcript to the bottom so
// the question is seen. That snap must not fire while the reader has scrolled up
// into history: they still get the floating jump control and a native
// notification, so the question cannot go unnoticed, but their reading position
// is theirs. Same rule the runStart guard applies to a new run.

const SESSION = 'active-session'

function context(overrides: Partial<GatewayEventContext> = {}): GatewayEventContext {
  const payload = { choices: ['yes', 'no'], question: 'Continue?', request_id: 'req-1' }

  return {
    deps: {
      activeSessionIdRef: { current: SESSION },
      sessionInterrupted: () => false,
      updateSessionState: (_sid, updater) => updater(createClientSessionState()),
      upsertToolCall: vi.fn()
    } as unknown as GatewayEventContext['deps'],
    event: { payload, session_id: SESSION, type: 'clarify.request' },
    explicitSid: SESSION,
    fromActiveSource: () => true,
    isActiveEvent: true,
    occurredAt: 1_700_000_100,
    payload: payload as GatewayEventContext['payload'],
    scheduleConfigRefresh: vi.fn(),
    sessionId: SESSION,
    ...overrides
  }
}

describe('clarify.request scroll guard', () => {
  afterEach(() => {
    resetThreadScroll()
    clearClarifyRequest()
  })

  it('snaps to the bottom when the reader is already there', () => {
    const scrolled = vi.fn()
    const off = onScrollToBottomRequest(scrolled, SESSION)
    setThreadAtBottom(true)

    expect(handleInputRequestEvent(context())).toBe(true)
    expect(scrolled).toHaveBeenCalledTimes(1)
    off()
  })

  it('leaves a scrolled-up reader where they are', () => {
    const scrolled = vi.fn()
    const off = onScrollToBottomRequest(scrolled, SESSION)
    setThreadAtBottom(false)

    expect(handleInputRequestEvent(context())).toBe(true)
    expect(scrolled).not.toHaveBeenCalled()
    off()
  })

  it('never snaps for a background session, scrolled or not', () => {
    const scrolled = vi.fn()
    const off = onScrollToBottomRequest(scrolled, SESSION)
    setThreadAtBottom(true)

    expect(
      handleInputRequestEvent(
        context({ deps: { ...context().deps, activeSessionIdRef: { current: 'other' } } as GatewayEventContext['deps'] })
      )
    ).toBe(true)
    expect(scrolled).not.toHaveBeenCalled()
    off()
  })

  it('applies the same guard to a batch (multi-question) request', () => {
    const scrolled = vi.fn()
    const off = onScrollToBottomRequest(scrolled, SESSION)
    const payload = {
      questions: [
        { choices: ['a', 'b'], qid: 'q0', question: 'First?' },
        { choices: ['c', 'd'], qid: 'q1', question: 'Second?' }
      ],
      request_id: 'req-batch'
    }
    const batch = (): GatewayEventContext =>
      context({ event: { payload, session_id: SESSION, type: 'clarify.request' }, payload: payload as GatewayEventContext['payload'] })

    setThreadAtBottom(false)
    expect(handleInputRequestEvent(batch())).toBe(true)
    // Prove the batch branch ran, so the guard under test is the batch one.
    expect(sessionClarifyRequest(SESSION).get()?.questions.length).toBe(2)
    expect(scrolled).not.toHaveBeenCalled()

    clearClarifyRequest()
    setThreadAtBottom(true)
    expect(handleInputRequestEvent(batch())).toBe(true)
    expect(scrolled).toHaveBeenCalledTimes(1)
    off()
  })
})
