import { afterEach, describe, expect, it, vi } from 'vitest'

import {
  $settledClarifyResults,
  type ClarifyRequest,
  clearClarifyRequest,
  rememberConfirmedClarify,
  setClarifyRequest
} from '@/store/clarify'

import { handleInputRequestEvent } from './input-requests'
import type { GatewayEventContext } from './types'

// The gateway replies to a card's last `clarify.lock`, then withdraws the request from every viewer with
// `request.cancel {reason: 'resolved'}`. The window that answered has already cleared its own card. An
// agent's clarify still gets its `tool.complete`; a card a plugin asked (`ctx.ask_choice` from a slash
// command) has no tool call, so the resolved cancel is the only signal left to settle that row.
function resolvedCancel(id: string) {
  const updateSessionState = vi.fn()
  const payload = { id, method: 'clarify', reason: 'resolved' }

  const ctx: GatewayEventContext = {
    deps: { flushQueuedDeltas: vi.fn(), updateSessionState } as unknown as GatewayEventContext['deps'],
    event: { payload, session_id: 's1', type: 'request.cancel' } as unknown as GatewayEventContext['event'],
    explicitSid: 's1',
    fromActiveSource: () => true,
    isActiveEvent: true,
    occurredAt: 1_700_000_000,
    payload: payload as unknown as GatewayEventContext['payload'],
    scheduleConfigRefresh: vi.fn(),
    sessionId: 's1'
  }

  return { ctx, updateSessionState }
}

const request: ClarifyRequest = {
  questions: [{ choices: ['yes', 'no'], multiSelect: false, qid: 'q0', question: 'Approve the plan?' }],
  requestId: 'srq-1',
  sessionId: 's1'
}

describe('request.cancel resolved for a card this window answered', () => {
  afterEach(() => {
    clearClarifyRequest()
    $settledClarifyResults.set({})
  })

  it('settles the row with the confirmed answer and lifts needs-input', () => {
    setClarifyRequest(request)
    rememberConfirmedClarify(request, { q0: 'yes' })
    // The card clears its own request once the last lock is accepted, before the cancel lands.
    clearClarifyRequest('srq-1', 's1')

    const { ctx, updateSessionState } = resolvedCancel('srq-1')

    expect(handleInputRequestEvent(ctx)).toBe(true)
    expect($settledClarifyResults.get()['srq-1']).toEqual({
      outcome: 'submitted',
      responses: [{ question: 'Approve the plan?', status: 'answered', user_response: 'yes' }]
    })
    expect(updateSessionState).toHaveBeenCalledTimes(1)
    expect(updateSessionState.mock.calls[0][1]({ needsInput: true })).toEqual({ needsInput: false })
  })

  it('lifts needs-input for an answer the composer already settled', () => {
    $settledClarifyResults.set({ 'srq-1': { outcome: 'submitted', responses: [] } })

    const { ctx, updateSessionState } = resolvedCancel('srq-1')

    expect(handleInputRequestEvent(ctx)).toBe(true)
    expect(updateSessionState).toHaveBeenCalledTimes(1)
    expect(updateSessionState.mock.calls[0][1]({ needsInput: true })).toEqual({ needsInput: false })
  })

  it('leaves a request this window never answered alone', () => {
    const { ctx, updateSessionState } = resolvedCancel('srq-other')

    expect(handleInputRequestEvent(ctx)).toBe(true)
    expect($settledClarifyResults.get()['srq-other']).toBeUndefined()
    expect(updateSessionState).not.toHaveBeenCalled()
  })
})
