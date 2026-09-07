import { beforeEach, describe, expect, it, vi } from 'vitest'

/**
 * Reproduction for #104764: after a turn interruption, a `clarify.request`
 * arrives while the session is still flagged interrupted and is silently
 * dropped — the one-shot event is consumed (`return true`) but never stored,
 * never replayed, so the Python side stays blocked on `clarify.respond`
 * forever and the user only sees a dead spinner card.
 */

const { setClarifyRequestMock } = vi.hoisted(() => ({
  setClarifyRequestMock: vi.fn(),
}))

vi.mock('@/store/clarify', () => ({
  $clarifyRequests: {},
  clearClarifyRequest: vi.fn(),
  normalizeChoices: (raw: unknown) => (Array.isArray(raw) ? raw : []),
  normalizeQuestions: (raw: unknown) => (Array.isArray(raw) ? raw : []),
  setClarifyRequest: setClarifyRequestMock,
  warnDroppedChoices: vi.fn(),
}))
vi.mock('@/store/gateway', () => ({ $gateway: {} }))
vi.mock('@/store/mcp-setup', () => ({ setMcpSetupRequest: vi.fn() }))
vi.mock('@/store/native-notifications', () => ({ dispatchNativeNotification: vi.fn() }))
vi.mock('@/store/prompts', () => ({
  receiveApprovalRequest: vi.fn(async () => undefined),
  setSecretRequest: vi.fn(),
  setSudoRequest: vi.fn(),
}))
vi.mock('@/store/thread-scroll', () => ({ requestScrollToBottom: vi.fn() }))
vi.mock('@/lib/chat-messages', () => ({
  restorePendingClarifyToolCall: () => ({ messages: [], streamId: null }),
  settlePendingClarifyToolCall: () => ({ messages: [], streamId: null }),
}))
vi.mock('@/app/session/hooks/use-session-actions/restore-pending-clarify', () => ({
  pendingClarifyToolPayload: (request: unknown) => request,
}))
vi.mock('@/i18n', () => ({ translateNow: () => 'input' }))

import { handleInputRequestEvent } from './input-requests'
import type { GatewayEventContext } from './types'

function context(overrides: { interrupted?: boolean } = {}): GatewayEventContext {
  const interrupted = overrides.interrupted ?? false
  return {
    deps: {
      activeSessionIdRef: { current: 's1' },
      sessionInterrupted: vi.fn(() => interrupted),
      updateSessionState: vi.fn(),
      upsertToolCall: vi.fn(),
    },
    event: { type: 'clarify.request' },
    explicitSid: undefined,
    fromActiveSource: () => true,
    isActiveEvent: true,
    occurredAt: 1_700_000_100,
    payload: {
      request_id: 'req-1',
      question: 'Delete this file?',
      choices: ['yes', 'no'],
    },
    scheduleConfigRefresh: vi.fn(),
    sessionId: 's1',
  } as unknown as GatewayEventContext
}

describe('handleInputRequestEvent clarify.request under an interrupted flag (#104764)', () => {
  beforeEach(() => {
    setClarifyRequestMock.mockClear()
  })

  it('stores the one-shot clarify.request even while the session is flagged interrupted', () => {
    // #104764: the Python side is already blocked on clarify.respond inside an
    // auto-continued turn. The request must be parked so the card renders the
    // question/choices and the user can answer — otherwise the turn strands
    // forever and the only exit is Stop, which loses the answer.
    const result = handleInputRequestEvent(context({ interrupted: true }))

    expect(result).toBe(true)
    expect(setClarifyRequestMock).toHaveBeenCalledTimes(1)
    const request = setClarifyRequestMock.mock.calls[0][0]
    expect(request.requestId).toBe('req-1')
    expect(request.question).toBe('Delete this file?')
    expect(request.choices).toEqual(['yes', 'no'])
  })

  it('stores the same event when the session is NOT interrupted (control)', () => {
    const result = handleInputRequestEvent(context({ interrupted: false }))

    expect(result).toBe(true)
    expect(setClarifyRequestMock).toHaveBeenCalledTimes(1)
    const request = setClarifyRequestMock.mock.calls[0][0]
    expect(request.requestId).toBe('req-1')
    expect(request.question).toBe('Delete this file?')
    expect(request.choices).toEqual(['yes', 'no'])
  })
})
