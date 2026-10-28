import { afterEach, describe, expect, it, vi } from 'vitest'

import { $notifications, clearNotifications } from '@/store/notifications'
import { $approvalRequests, clearApprovalRequest, setApprovalRequest } from '@/store/prompts'

import { handleInputRequestEvent } from './input-requests'
import type { GatewayEventContext } from './types'

// A canonical gateway fans one approval out to every attached window; whichever window
// answers, the authority publishes `approval.settled {prompt_id}`. Every other window's
// Run/Reject bar must come down then, not stay parked on a prompt nobody can answer.
function settled(
  promptId: string,
  extra: { reason?: string } = {},
  updateSessionState: ReturnType<typeof vi.fn> = vi.fn()
): GatewayEventContext {
  const payload = { prompt_id: promptId, execution_generation: 3, ...extra }

  return {
    deps: { flushQueuedDeltas: vi.fn(), updateSessionState } as unknown as GatewayEventContext['deps'],
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
  afterEach(() => {
    clearApprovalRequest('s1')
    clearNotifications()
  })

  it('clears the bar a peer window answered and leaves other prompts alone', () => {
    setApprovalRequest({ sessionId: 's1', command: 'rm a', description: '', requestId: 'p-1', serverRequestId: 'p-1' })
    setApprovalRequest({ sessionId: 's1', command: 'rm b', description: '', requestId: 'p-2', serverRequestId: 'p-2' })
    const updateSessionState = vi.fn()

    expect(handleInputRequestEvent(settled('p-1', {}, updateSessionState))).toBe(true)
    expect($approvalRequests.get()['s1']?.requestId).toBe('p-2')
    expect(updateSessionState).not.toHaveBeenCalled()
    expect($notifications.get()).toHaveLength(0)
    clearApprovalRequest('s1', 'p-2')
    expect($approvalRequests.get()['s1']).toBeUndefined()
  })

  // The shared gateway expires an unanswered approval with `reason: "timeout"`. The
  // bar must not just vanish: the transcript says the command did not run, once.
  it('timeout: says the command did not run, once, even if the settle is replayed', () => {
    setApprovalRequest({
      sessionId: 's1',
      command: 'rm -rf build',
      description: '',
      requestId: 'p-1',
      serverRequestId: 'p-1'
    })
    let state = { messages: [] as { id: string; role: string; parts: { text: string }[] }[] }

    const updateSessionState = vi.fn((_: string, updater: (s: typeof state) => typeof state) => {
      state = updater(state)
    })

    expect(handleInputRequestEvent(settled('p-1', { reason: 'timeout' }, updateSessionState))).toBe(true)
    expect($approvalRequests.get()['s1']).toBeUndefined()
    expect(state.messages).toHaveLength(1)
    expect(state.messages[0].role).toBe('system')
    expect(state.messages[0].parts[0].text).toMatch(/timed out.*not run/i)
    expect($notifications.get()[0]?.kind).toBe('warning')

    // A replayed settle finds no parked card and adds nothing.
    handleInputRequestEvent(settled('p-1', { reason: 'timeout' }, updateSessionState))
    expect(state.messages).toHaveLength(1)
  })
})
