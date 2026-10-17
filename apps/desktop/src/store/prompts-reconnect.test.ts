import { JsonRpcGatewayError } from '@hermes/shared'
import { afterEach, expect, it } from 'vitest'

import { approvalReplaySessionId } from '@/lib/gateway-events'

import { clearAllPrompts, replayPendingApproval, sessionApprovalRequests } from './prompts'
import { isSessionGone, resetBackgroundPollingGuard, resetBackgroundPollingGuardAfterRebind } from './session-gone-latch'

afterEach(() => {
  clearAllPrompts()
  resetBackgroundPollingGuard()
})

it('waits for session rebind before replaying approvals across repeated reconnects', async () => {
  const queried: string[] = []
  let liveRuntime = 'runtime-0'

  const gateway = {
    request: async (method: string, params: Record<string, unknown>) => {
      if (method === 'approval.pending') {
        queried.push(String(params.session_id))

        if (params.session_id !== liveRuntime) {
          throw new JsonRpcGatewayError('session not found', { code: 4001 })
        }

        return { approvals: [{ request_id: `approval-${liveRuntime}`, command: 'echo test', description: 'test' }] }
      }

      return { acknowledged: true }
    }
  }

  for (let reconnect = 1; reconnect <= 3; reconnect++) {
    const cachedRuntime = liveRuntime
    // The disconnected runtime was reaped before the next socket became ready.
    liveRuntime = ''
    resetBackgroundPollingGuard()
    await replayPendingApproval(
      gateway,
      approvalReplaySessionId('gateway.ready', cachedRuntime, { explicit: false, isGone: isSessionGone })
    )

    liveRuntime = `runtime-${reconnect}`
    resetBackgroundPollingGuardAfterRebind('session.resume', { session_id: 'stored-session' }, { session_id: liveRuntime })
    await replayPendingApproval(
      gateway,
      approvalReplaySessionId('session.info', liveRuntime, { explicit: true, isGone: isSessionGone })
    )
    expect(sessionApprovalRequests(liveRuntime).get().map(request => request.requestId)).toEqual([
      `approval-${liveRuntime}`
    ])
  }

  expect(queried).toEqual(['runtime-1', 'runtime-2', 'runtime-3'])
})
