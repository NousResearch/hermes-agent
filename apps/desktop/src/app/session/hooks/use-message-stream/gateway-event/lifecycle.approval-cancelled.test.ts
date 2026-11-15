import { afterEach, describe, expect, it, vi } from 'vitest'

import { $approvalRequests, clearApprovalRequest, setApprovalRequest } from '@/store/prompts'

import { handleLifecycleEvent } from './lifecycle'
import type { GatewayEventContext } from './types'

// #106678: interrupt/reap/teardown deny-resolves the backend approval queue
// silently. The approval.cancelled broadcast is the client's only signal —
// without it a parked Run/Reject bar keeps offering buttons that answer
// `resolved: 0` against a prompt the backend already dropped.
function cancelledContext(payload: Record<string, unknown>): GatewayEventContext {
  return {
    deps: {} as GatewayEventContext['deps'],
    event: { payload, type: 'approval.cancelled' },
    explicitSid: 's1',
    fromActiveSource: () => true,
    isActiveEvent: true,
    occurredAt: 1_700_000_000,
    payload: payload as GatewayEventContext['payload'],
    scheduleConfigRefresh: vi.fn(),
    sessionId: 's1'
  }
}

function parkApproval(requestId: string, sessionId = 's1') {
  setApprovalRequest({
    sessionId,
    command: 'rm -rf build',
    description: '',
    requestId,
    serverRequestId: `srv-${requestId}`
  })
}

describe('approval.cancelled lifecycle event', () => {
  afterEach(() => {
    clearApprovalRequest('s1')
  })

  it('clears the parked prompt for each named request id', () => {
    parkApproval('req-1')

    expect(
      handleLifecycleEvent(
        cancelledContext({
          cancelled_count: 1,
          reason: 'ws_orphan_reap',
          request_ids: ['req-1'],
          session_id: 's1',
          stored_session_id: '20260909_120000_aaaaaa'
        })
      )
    ).toBe(true)
    expect($approvalRequests.get()['s1']).toBeUndefined()
  })

  it('spares a newer prompt armed under a different request id on the same session', () => {
    parkApproval('req-old')

    handleLifecycleEvent(
      cancelledContext({
        cancelled_count: 1,
        reason: 'ws_orphan_reap',
        request_ids: ['req-gone'],
        session_id: 's1',
        stored_session_id: '20260909_120000_aaaaaa'
      })
    )

    // The id mismatch makes clear() a no-op — a cancelled approval can never
    // wipe a prompt re-armed by a live turn on the same session.
    expect($approvalRequests.get()['s1']?.requestId).toBe('req-old')
  })

  it('without correlation ids, drops the session prompt wholesale', () => {
    parkApproval('req-1')

    handleLifecycleEvent(
      cancelledContext({
        cancelled_count: 1,
        reason: 'interrupt',
        request_ids: [],
        session_id: 's1',
        stored_session_id: '20260909_120000_aaaaaa'
      })
    )

    expect($approvalRequests.get()['s1']).toBeUndefined()
  })

  it('ignores a broadcast for another session', () => {
    parkApproval('req-1')

    handleLifecycleEvent(
      cancelledContext({
        cancelled_count: 1,
        reason: 'interrupt',
        request_ids: ['req-1'],
        session_id: 'other-runtime',
        stored_session_id: 'other'
      })
    )

    expect($approvalRequests.get()['s1']?.requestId).toBe('req-1')
  })
})
