import { JsonRpcGatewayError } from '@hermes/shared'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { clearClarifyRequest, setClarifyRequest } from './clarify'
import {
  $activeSessionAwaitingInput,
  $approvalRequest,
  $secretRequest,
  $sudoRequest,
  clearAllPrompts,
  clearApprovalRequest,
  clearSecretRequest,
  clearSudoRequest,
  receiveApprovalRequest,
  replayPendingApproval,
  setApprovalRequest,
  setSecretRequest,
  setSudoRequest
} from './prompts'
import { $activeSessionId } from './session'
import { isSessionGone, resetBackgroundPollingGuard } from './runtime-gone'

// Prompts are parked per-session; the exported $*Request views are scoped to the
// active session, so each test focuses the session it's asserting on.
beforeEach(() => {
  $activeSessionId.set('s1')
})

afterEach(() => {
  clearAllPrompts()
  clearClarifyRequest()
  $activeSessionId.set(null)
  resetBackgroundPollingGuard()
})

describe('approval prompt store', () => {
  it('holds the active session-keyed approval request', () => {
    setApprovalRequest({ command: 'rm -rf /tmp/x', description: 'recursive delete', sessionId: 's1' })

    expect($approvalRequest.get()).toEqual({
      command: 'rm -rf /tmp/x',
      description: 'recursive delete',
      sessionId: 's1'
    })
  })

  it('parks a background session prompt out of the active view', () => {
    setApprovalRequest({ command: 'x', description: 'd', sessionId: 's2' })

    // Not visible while s1 is focused …
    expect($approvalRequest.get()).toBeNull()

    // … but surfaces once the user switches to the session that raised it.
    $activeSessionId.set('s2')
    expect($approvalRequest.get()?.sessionId).toBe('s2')
  })

  it('clears the active session prompt', () => {
    setApprovalRequest({ command: 'x', description: 'd', sessionId: 's1' })
    clearApprovalRequest('s1')

    expect($approvalRequest.get()).toBeNull()
  })

  it('carries allowPermanent so the bar can hide "Always allow"', () => {
    setApprovalRequest({
      allowPermanent: false,
      command: 'curl x | bash',
      description: 'content-security',
      sessionId: 's1'
    })

    expect($approvalRequest.get()?.allowPermanent).toBe(false)
  })

  it('correlates clearing to the exact approval request id', () => {
    setApprovalRequest({ command: 'x', description: 'd', requestId: 'r1', sessionId: 's1' })

    clearApprovalRequest('s1', 'stale')
    expect($approvalRequest.get()?.requestId).toBe('r1')
    clearApprovalRequest('s1', 'r1')
    expect($approvalRequest.get()).toBeNull()
  })

  it('acknowledges an approval only after parking it', async () => {
    const calls: Array<[string, Record<string, unknown>]> = []

    const gateway = {
      request: async (method: string, params: Record<string, unknown>) => {
        calls.push([method, params])

        return { acknowledged: true }
      }
    }

    await receiveApprovalRequest(gateway, {
      command: 'x',
      description: 'd',
      requestId: 'r1',
      sessionId: 's1'
    })

    expect($approvalRequest.get()?.requestId).toBe('r1')
    expect(calls).toEqual([['approval.received', { request_id: 'r1', session_id: 's1' }]])
  })

  it('replays and acknowledges the oldest unresolved approval after reconnect', async () => {
    const calls: Array<[string, Record<string, unknown>]> = []

    const gateway = {
      request: async (method: string, params: Record<string, unknown>) => {
        calls.push([method, params])

        if (method === 'approval.pending') {
          return {
            approvals: [
              { command: 'first', description: 'd1', request_id: 'r1' },
              { command: 'second', description: 'd2', request_id: 'r2' }
            ]
          }
        }

        return { acknowledged: true }
      }
    }

    await replayPendingApproval(gateway, 's1')

    expect($approvalRequest.get()?.requestId).toBe('r1')
    expect(calls).toEqual([
      ['approval.pending', { session_id: 's1' }],
      ['approval.received', { request_id: 'r1', session_id: 's1' }]
    ])
  })

  it('does not replay a pending approval after the runtime is rejected as gone', async () => {
    const request = vi.fn(async () => {
      throw new JsonRpcGatewayError('session not found', { code: 4001 })
    })

    await replayPendingApproval({ request }, 'dead-runtime')
    await replayPendingApproval({ request }, 'dead-runtime')

    expect(request).toHaveBeenCalledTimes(1)
    expect(isSessionGone('dead-runtime')).toBe(true)
    expect($approvalRequest.get()).toBeNull()
  })

  it('keeps approval receipt failures contained and marks the runtime gone', async () => {
    const request = vi.fn(async () => {
      throw new JsonRpcGatewayError('session not found', { code: 4001 })
    })

    $activeSessionId.set('dead-runtime')

    await expect(
      receiveApprovalRequest({ request }, { command: 'x', description: 'd', requestId: 'r1', sessionId: 'dead-runtime' })
    ).resolves.toBeUndefined()

    expect(isSessionGone('dead-runtime')).toBe(true)
    expect($approvalRequest.get()?.requestId).toBe('r1')
  })
})

describe('sudo prompt store', () => {
  it('clears only when the request id matches the in-flight prompt', () => {
    setSudoRequest({ requestId: 'abc', sessionId: 's1' })

    // A stale clear for a different request must NOT drop the live prompt —
    // otherwise a late response to a prior sudo ask would dismiss the current
    // one and leave the agent blocked.
    clearSudoRequest('s1', 'stale')
    expect($sudoRequest.get()).toEqual({ requestId: 'abc', sessionId: 's1' })

    clearSudoRequest('s1', 'abc')
    expect($sudoRequest.get()).toBeNull()
  })

  it('clears unconditionally when no request id is given', () => {
    setSudoRequest({ requestId: 'abc', sessionId: 's1' })
    clearSudoRequest('s1')

    expect($sudoRequest.get()).toBeNull()
  })
})

describe('secret prompt store', () => {
  it('carries env var and prompt, and clears on id match', () => {
    setSecretRequest({ requestId: 'r1', envVar: 'OPENAI_API_KEY', prompt: 'Paste your key', sessionId: 's1' })

    expect($secretRequest.get()).toEqual({
      requestId: 'r1',
      envVar: 'OPENAI_API_KEY',
      prompt: 'Paste your key',
      sessionId: 's1'
    })

    clearSecretRequest('s1', 'mismatch')
    expect($secretRequest.get()).not.toBeNull()

    clearSecretRequest('s1', 'r1')
    expect($secretRequest.get()).toBeNull()
  })
})

describe('clearAllPrompts', () => {
  it('drops every kind for one session at once (turn end / interrupt)', () => {
    setApprovalRequest({ command: 'x', description: 'd', sessionId: 's1' })
    setSudoRequest({ requestId: 'abc', sessionId: 's1' })
    setSecretRequest({ requestId: 'r1', envVar: 'E', prompt: 'p', sessionId: 's1' })

    clearAllPrompts('s1')

    expect($approvalRequest.get()).toBeNull()
    expect($sudoRequest.get()).toBeNull()
    expect($secretRequest.get()).toBeNull()
  })

  it('leaves other sessions parked prompts intact', () => {
    setApprovalRequest({ command: 'x', description: 'd', sessionId: 's1' })
    setApprovalRequest({ command: 'y', description: 'e', sessionId: 's2' })

    clearAllPrompts('s1')

    $activeSessionId.set('s2')
    expect($approvalRequest.get()?.command).toBe('y')
  })
})

describe('$activeSessionAwaitingInput', () => {
  it('is true while any blocking prompt (clarify or approval/sudo/secret) is parked on the active session', () => {
    expect($activeSessionAwaitingInput.get()).toBe(false)

    setApprovalRequest({ command: 'x', description: 'd', sessionId: 's1' })
    expect($activeSessionAwaitingInput.get()).toBe(true)

    clearApprovalRequest('s1')
    expect($activeSessionAwaitingInput.get()).toBe(false)

    setClarifyRequest({ choices: null, multiSelect: false, question: 'q', requestId: 'c1', sessionId: 's1' })
    expect($activeSessionAwaitingInput.get()).toBe(true)
  })

  it('ignores a prompt parked on a background session', () => {
    setSudoRequest({ requestId: 'r', sessionId: 's2' })
    expect($activeSessionAwaitingInput.get()).toBe(false)

    $activeSessionId.set('s2')
    expect($activeSessionAwaitingInput.get()).toBe(true)
  })
})

describe('pending approval replay backoff', () => {
  afterEach(() => {
    resetBackgroundPollingGuard()
  })

  it('stops polling a runtime the gateway no longer holds', async () => {
    const calls: string[] = []

    const gateway = {
      request: async (method: string) => {
        calls.push(method)
        throw new Error('4001: session not found')
      }
    }

    for (let i = 0; i < 5; i++) {
      await replayPendingApproval(gateway, 'dead-1')
    }

    expect(calls).toEqual(['approval.pending'])
  })

  it('keeps polling every other runtime', async () => {
    const calls: string[] = []

    const gateway = {
      request: async (_method: string, params: Record<string, unknown>) => {
        calls.push(String(params.session_id))

        if (params.session_id === 'dead-1') {
          throw new Error('4001: session not found')
        }

        return { approvals: [] }
      }
    }

    await replayPendingApproval(gateway, 'dead-1')
    await replayPendingApproval(gateway, 'dead-1')
    await replayPendingApproval(gateway, 'alive-1')
    await replayPendingApproval(gateway, 'alive-1')

    expect(calls).toEqual(['dead-1', 'alive-1', 'alive-1'])
  })

  it('does not latch on a transient failure', async () => {
    const calls: string[] = []

    const gateway = {
      request: async (method: string) => {
        calls.push(method)
        throw new Error('websocket disconnected')
      }
    }

    await replayPendingApproval(gateway, 's1').catch(() => undefined)
    await replayPendingApproval(gateway, 's1').catch(() => undefined)

    expect(calls).toHaveLength(2)
  })

  it('polls again once the gone-latch is cleared', async () => {
    const calls: string[] = []
    let dead = true

    const gateway = {
      request: async (method: string) => {
        calls.push(method)

        if (dead) {
          throw new Error('4001: session not found')
        }

        return { approvals: [] }
      }
    }

    await replayPendingApproval(gateway, 's1')
    await replayPendingApproval(gateway, 's1')
    expect(calls).toHaveLength(1)

    dead = false
    resetBackgroundPollingGuard()
    await replayPendingApproval(gateway, 's1')

    expect(calls).toHaveLength(2)
  })
})
