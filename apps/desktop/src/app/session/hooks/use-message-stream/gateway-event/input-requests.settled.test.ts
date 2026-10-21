import { afterEach, describe, expect, it, vi } from 'vitest'

import { $approvalRequests, clearApprovalRequest, setApprovalRequest } from '@/store/prompts'

import { handleInputRequestEvent } from './input-requests'
import type { GatewayEventContext } from './types'

// A canonical gateway fans one approval out to every attached window; whichever window
// answers, the authority publishes `approval.settled {prompt_id}`. Every other window's
// Run/Reject bar must come down then, not stay parked on a prompt nobody can answer.
function settled(promptId: string): GatewayEventContext {
  const payload = { prompt_id: promptId, execution_generation: 3 }

  return {
    deps: { flushQueuedDeltas: vi.fn(), updateSessionState: vi.fn() } as unknown as GatewayEventContext['deps'],
    event: { payload, session_id: 's1', type: 'approval.settled' } as unknown as GatewayEventContext['event'],
    explicitSid: 's1',
    fromActiveSource: () => true,
    isActiveEvent: true,
    occurredAt: 1_700_000_000,
    payload: payload as unknown as GatewayEventContext['payload'],
    scheduleConfigRefresh: vi.fn(),
    sessionId: 's1'
  }
}

describe('canonical approval.settled', () => {
  afterEach(() => clearApprovalRequest('s1'))

  it('clears the bar a peer window answered and leaves other prompts alone', () => {
    setApprovalRequest({ sessionId: 's1', command: 'rm a', description: '', requestId: 'p-1', serverRequestId: 'p-1' })
    setApprovalRequest({ sessionId: 's1', command: 'rm b', description: '', requestId: 'p-2', serverRequestId: 'p-2' })

    expect(handleInputRequestEvent(settled('p-1'))).toBe(true)
    expect($approvalRequests.get()['s1']?.requestId).toBe('p-2')
    clearApprovalRequest('s1', 'p-2')
    expect($approvalRequests.get()['s1']).toBeUndefined()
  })
})
