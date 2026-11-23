import { afterEach, describe, expect, it, vi } from 'vitest'

import { $notifications, clearNotifications } from '@/store/notifications'
import { $approvalRequests, clearApprovalRequest, setApprovalRequest } from '@/store/prompts'
import { $routeRequest } from '@/store/recovery-requests'

import { handleInputRequestEvent } from './input-requests'
import type { GatewayEventContext } from './types'

// When the server withdraws an unanswered approval on timeout, the Run/Reject
// bar disappears and the tool row then shows a model-facing "BLOCKED …" result.
// The transcript must gain a human system line that says what happened and
// where the wait is configured; a cancel for any other reason stays silent.
function context(reason: string, updateSessionState: ReturnType<typeof vi.fn>): GatewayEventContext {
  const payload = { id: 'srv-1', method: 'approval', reason }

  return {
    deps: { flushQueuedDeltas: vi.fn(), updateSessionState } as unknown as GatewayEventContext['deps'],
    event: { payload, session_id: 's1', type: 'request.cancel' },
    explicitSid: 's1',
    fromActiveSource: () => true,
    isActiveEvent: true,
    occurredAt: 1_700_000_000,
    payload: payload as GatewayEventContext['payload'],
    scheduleConfigRefresh: vi.fn(),
    sessionId: 's1'
  }
}

function parkApproval() {
  setApprovalRequest({
    sessionId: 's1',
    command: 'rm -rf build',
    description: '',
    requestId: 'req-1',
    serverRequestId: 'srv-1'
  })
}

describe('approval request.cancel', () => {
  afterEach(() => {
    clearApprovalRequest('s1')
    clearNotifications()
  })

  it('timeout: tears the bar down and appends a plain system line with a Safety settings action', () => {
    parkApproval()

    const updateSessionState = vi.fn((_: string, updater: (s: { messages: unknown[] }) => unknown) =>
      updater({ messages: [] })
    )

    expect(handleInputRequestEvent(context('timeout', updateSessionState))).toBe(true)
    expect($approvalRequests.get()['s1']).toBeUndefined()

    const next = updateSessionState.mock.results[0]?.value as {
      messages: { role: string; parts: { text: string }[] }[]
    }

    expect(next.messages).toHaveLength(1)
    expect(next.messages[0].role).toBe('system')
    expect(next.messages[0].parts[0].text).toMatch(/timed out/i)
    expect(next.messages[0].parts[0].text).not.toMatch(/BLOCKED|Do NOT/)

    const toast = $notifications.get()[0]
    toast?.action?.onClick()
    expect($routeRequest.get()?.path).toBe('/settings?tab=config:safety')
  })

  it('timeout of a folder card: the line says access to the folder was not granted', () => {
    setApprovalRequest({
      sessionId: 's1',
      command: '',
      description: 'Allow read access to C:\\ref',
      kind: 'access',
      requestId: 'req-1',
      serverRequestId: 'srv-1',
      target: { access: 'read', path: 'C:\\ref' }
    })

    const updateSessionState = vi.fn((_: string, updater: (s: { messages: unknown[] }) => unknown) =>
      updater({ messages: [] })
    )

    expect(handleInputRequestEvent(context('timeout', updateSessionState))).toBe(true)

    const next = updateSessionState.mock.results[0]?.value as { messages: { parts: { text: string }[] }[] }
    const line = next.messages[0].parts[0].text

    expect(line).toContain('access to the folder C:\\ref was not granted')
    expect(line).not.toMatch(/command|Settings/i)

    // The card's sender owns the wait, so the toast does not offer the approval timeout setting.
    expect($notifications.get()[0]?.message).toBe(line)
    expect($notifications.get()[0]?.action).toBeUndefined()
  })

  it('timeout of a site card: the line names the site as the card did', () => {
    setApprovalRequest({
      sessionId: 's1',
      command: '',
      description: 'Allow the sandbox to reach Web search (api.keenable.ai)',
      kind: 'network',
      requestId: 'req-1',
      serverRequestId: 'srv-1',
      target: { host: 'api.keenable.ai', label: 'Web search (api.keenable.ai)' }
    })

    const updateSessionState = vi.fn((_: string, updater: (s: { messages: unknown[] }) => unknown) =>
      updater({ messages: [] })
    )

    expect(handleInputRequestEvent(context('timeout', updateSessionState))).toBe(true)

    const next = updateSessionState.mock.results[0]?.value as { messages: { parts: { text: string }[] }[] }

    expect(next.messages[0].parts[0].text).toContain('access to Web search (api.keenable.ai) was not granted')
    expect(next.messages[0].parts[0].text).not.toMatch(/command/i)
  })

  it('withdraws a queued request without dropping the front approval', () => {
    setApprovalRequest({
      sessionId: 's1',
      command: 'first',
      description: '',
      requestId: 'front',
      serverRequestId: 'srv-front'
    })
    parkApproval()
    const updateSessionState = vi.fn()

    expect(handleInputRequestEvent(context('answered', updateSessionState))).toBe(true)
    expect($approvalRequests.get()['s1']?.requestId).toBe('front')
    clearApprovalRequest('s1', 'front')
    expect($approvalRequests.get()['s1']).toBeUndefined()
    expect(updateSessionState).not.toHaveBeenCalled()
  })

  it('answered elsewhere: tears the bar down silently', () => {
    parkApproval()
    const updateSessionState = vi.fn()

    expect(handleInputRequestEvent(context('answered', updateSessionState))).toBe(true)
    expect($approvalRequests.get()['s1']).toBeUndefined()
    expect(updateSessionState).not.toHaveBeenCalled()
    expect($notifications.get()).toHaveLength(0)
  })
})
