// The gateway `error` event (tui_gateway/prompt_turn.py) carries only a
// message — no error_surface. The dispatcher must still classify the two
// refusals it can mean so the card and toast read like a classified turn:
// plain words up front, raw text demoted to the detail line, and Retry gone
// where retrying reproduces the failure.
import { afterEach, describe, expect, it, vi } from 'vitest'

import { $notifications } from '@/store/notifications'

import { handleStatusEvent } from './status'
import type { GatewayEventContext } from './types'

vi.mock('@/store/native-notifications', () => ({ dispatchNativeNotification: vi.fn() }))
vi.mock('@/store/onboarding', () => ({ requestDesktopOnboarding: vi.fn() }))

const OWNED_REFUSAL =
  'Session 20260909_095312_6b93f5 already has a live owner (tui, pid 32977, lease age 22m). ' +
  'Attach through a compatible owner, or close the session in its owning surface before resuming here.'

function errorContext(message: string) {
  const failAssistantMessage = vi.fn()
  const payload = { message } as GatewayEventContext['payload']

  const ctx: GatewayEventContext = {
    deps: {
      compactedTurnRef: { current: new Set<string>() },
      failAssistantMessage,
      flushQueuedDeltas: vi.fn(),
      hydrateFromStoredSession: vi.fn(),
      queryClient: { invalidateQueries: vi.fn() },
      sessionStateByRuntimeIdRef: { current: new Map() },
      updateSessionState: vi.fn()
    } as unknown as GatewayEventContext['deps'],
    event: { payload, session_id: 'sess-1', type: 'error' },
    explicitSid: 'sess-1',
    fromActiveSource: () => true,
    isActiveEvent: false,
    occurredAt: 1_700_000_100,
    payload,
    scheduleConfigRefresh: vi.fn(),
    sessionId: 'sess-1'
  }

  return { ctx, failAssistantMessage }
}

afterEach(() => {
  $notifications.set([])
})

describe('gateway `error` event → error card + toast', () => {
  it('stamps SESSION_NOT_OWNED on a live-owner refusal so the card drops Retry', () => {
    const { ctx, failAssistantMessage } = errorContext(OWNED_REFUSAL)

    expect(handleStatusEvent(ctx)).toBe(true)

    expect(failAssistantMessage).toHaveBeenCalledWith(
      'sess-1',
      OWNED_REFUSAL,
      1_700_000_100,
      expect.objectContaining({ code: 'SESSION_NOT_OWNED', retryable: false })
    )
  })

  it('toasts the plain explanation, not the lease jargon, and keeps the raw text as detail', () => {
    const { ctx } = errorContext(OWNED_REFUSAL)

    handleStatusEvent(ctx)

    const toast = $notifications.get()[0]

    expect(toast.message).toBeTruthy()
    expect(toast.message).not.toMatch(/lease|pid|live owner/i)
    expect(toast.detail).toBe(OWNED_REFUSAL)
    expect(toast.action).toBeUndefined()
  })

  it("keeps the server's own plain copy as the toast message when no code was recovered", () => {
    // tui_gateway/user_messages.py already writes actionable sentences for
    // pre-turn failures; the generic "couldn't finish" gloss must not bury them.
    const serverCopy = 'Hermes could not start the assistant for this chat. Check your model settings and try again.'
    const { ctx, failAssistantMessage } = errorContext(serverCopy)

    handleStatusEvent(ctx)

    expect(failAssistantMessage).toHaveBeenCalledWith('sess-1', serverCopy, 1_700_000_100, null)
    const toast = $notifications.get()[0]
    expect(toast.message).toBe(serverCopy)
    expect(toast.detail).toBeUndefined()
  })
})

function warnContext(text: string) {
  const updateSessionState = vi.fn()

  const payload = { kind: 'warn', text } as GatewayEventContext['payload']

  const ctx: GatewayEventContext = {
    deps: {
      compactedTurnRef: { current: new Set<string>() },
      failAssistantMessage: vi.fn(),
      flushQueuedDeltas: vi.fn(),
      hydrateFromStoredSession: vi.fn(),
      queryClient: { invalidateQueries: vi.fn() },
      sessionStateByRuntimeIdRef: { current: new Map() },
      updateSessionState
    } as unknown as GatewayEventContext['deps'],
    event: { payload, session_id: 'sess-1', type: 'status.update' },
    explicitSid: 'sess-1',
    fromActiveSource: () => true,
    isActiveEvent: false,
    occurredAt: 1_700_000_100,
    payload,
    scheduleConfigRefresh: vi.fn(),
    sessionId: 'sess-1'
  }

  return { ctx, updateSessionState }
}

describe('status.update kind=warn → persistent transcript line', () => {
  // The compressor's context-lockout warning (#101889 companion) rides the
  // status rail; the desktop previously had no warn branch and dropped it —
  // a session that could no longer compress kept slowing, warning only in
  // agent.log. It must land as a persistent system line, deduped by id.

  const LOCKOUT =
    "⚠ Context lockout: this conversation can't be compressed and will keep slowing — /compact to compress history now or start a new chat."

  it('appends a system message instead of dropping the warn silently', () => {
    const { ctx, updateSessionState } = warnContext(LOCKOUT)

    expect(handleStatusEvent(ctx)).toBe(true)

    const updater = updateSessionState.mock.calls[0][1]
    const next = updater({ messages: [] })

    expect(next.messages).toHaveLength(1)
    expect(next.messages[0].role).toBe('system')
    expect(next.messages[0].id).toBe(`status-warn:${LOCKOUT.slice(0, 120)}`)
    expect(next.messages[0].parts[0].text).toBe(LOCKOUT)
  })

  it('collapses a warn that repeats every turn into one line', () => {
    const { ctx, updateSessionState } = warnContext(LOCKOUT)

    handleStatusEvent(ctx)

    const updater = updateSessionState.mock.calls[0][1]
    const first = updater({ messages: [] })
    const second = updater(first)

    expect(second.messages).toHaveLength(1)
  })

  it('ignores an empty warn payload', () => {
    const { ctx, updateSessionState } = warnContext('   ')

    handleStatusEvent(ctx)

    expect(updateSessionState).not.toHaveBeenCalled()
  })
})
