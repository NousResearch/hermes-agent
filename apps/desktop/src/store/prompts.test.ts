import { JsonRpcGatewayError, type PendingApproval, type RpcMethods } from '@hermes/shared'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { clearClarifyRequest, setClarifyRequest } from './clarify'
import {
  $activeSessionAwaitingInput,
  $approvalRequest,
  $secretRequest,
  $sudoRequest,
  type ApprovalGateway,
  clearAllPrompts,
  clearApprovalRequest,
  clearSecretRequest,
  clearSudoRequest,
  receiveApprovalRequest,
  replayPendingApproval,
  sessionApprovalRequests,
  setApprovalRequest,
  setSecretRequest,
  setSudoRequest
} from './prompts'
import { isSessionGone, resetBackgroundPollingGuard } from './runtime-gone'
import { $activeSessionId, setActiveSessionId } from './session'

type RpcCall = [keyof RpcMethods, RpcMethods[keyof RpcMethods]['params']]

type Replies = Partial<{ [M in keyof RpcMethods]: RpcMethods[M]['result'] }>

/** A gateway that answers from a reply table and records every call. `fail`
 *  turns a call into a rejection; `gate` holds every answer until it settles. */
function fakeGateway(
  replies: Replies,
  fail?: (method: keyof RpcMethods, params: RpcMethods[keyof RpcMethods]['params']) => Error | undefined,
  gate: Promise<void> = Promise.resolve()
) {
  const calls: RpcCall[] = []

  const gateway: ApprovalGateway = {
    request: async <M extends keyof RpcMethods>(
      method: M,
      params: RpcMethods[M]['params']
    ): Promise<RpcMethods[M]['result']> => {
      calls.push([method, params])
      await gate

      const error = fail?.(method, params)

      if (error) {
        throw error
      }

      const reply = replies[method]

      if (reply === undefined) {
        throw new Error(`unexpected RPC ${method}`)
      }

      // SAFETY: `replies` is keyed by method; relating Replies[M] to RpcMethods[M]['result'] generically
      // makes tsc compare every method's result (37 s / 4 GB on this file alone).
      return reply as RpcMethods[M]['result']
    }
  }

  return { calls, gateway }
}

const pendingApproval = (fields: Pick<PendingApproval, 'command' | 'description' | 'request_id'>): PendingApproval => ({
  allow_permanent: null,
  allow_session: null,
  choices: null,
  pattern_key: null,
  pattern_keys: null,
  smart_denied: null,
  tool_name: null,
  ...fields
})

const sessionNotFound = () => new JsonRpcGatewayError('session not found', { code: 4001 })

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
    const { calls, gateway } = fakeGateway({ 'approval.received': { acknowledged: true } })

    await receiveApprovalRequest(gateway, {
      command: 'x',
      description: 'd',
      requestId: 'r1',
      sessionId: 's1'
    })

    expect($approvalRequest.get()?.requestId).toBe('r1')
    expect(calls).toEqual([['approval.received', { request_id: 'r1', session_id: 's1' }]])
  })

  it('replays and acknowledges every unresolved approval after reconnect', async () => {
    const { calls, gateway } = fakeGateway({
      'approval.pending': {
        approvals: [
          pendingApproval({ command: 'first', description: 'd1', request_id: 'r1' }),
          pendingApproval({ command: 'second', description: 'd2', request_id: 'r2' })
        ]
      },
      'approval.received': { acknowledged: true }
    })

    await replayPendingApproval(gateway, 's1')

    expect($approvalRequest.get()?.requestId).toBe('r1')
    expect(calls).toEqual([
      ['approval.pending', { session_id: 's1' }],
      ['approval.received', { request_id: 'r1', session_id: 's1' }],
      ['approval.received', { request_id: 'r2', session_id: 's1' }]
    ])
    expect(
      sessionApprovalRequests('s1')
        .get()
        .map(request => request.requestId)
    ).toEqual(['r1', 'r2'])
    clearApprovalRequest('s1', 'r1')
    expect($approvalRequest.get()?.requestId).toBe('r2')
  })

  it('preserves live server request routing for every queued replay entry', async () => {
    for (const id of ['r1', 'r2']) {
      await receiveApprovalRequest(null, {
        command: id,
        description: id,
        requestId: id,
        serverRequestId: `srv-${id}`,
        sessionId: 's1'
      })
    }

    await replayPendingApproval(
      fakeGateway({
        'approval.pending': {
          approvals: [
            pendingApproval({ command: 'r1', description: 'r1', request_id: 'r1' }),
            pendingApproval({ command: 'r2', description: 'r2', request_id: 'r2' })
          ]
        },
        'approval.received': { acknowledged: true }
      }).gateway,
      's1'
    )
    expect(
      sessionApprovalRequests('s1')
        .get()
        .map(request => request.serverRequestId)
    ).toEqual(['srv-r1', 'srv-r2'])
  })

  it('deduplicates queued ids and rejects a replay that races an exact response', async () => {
    const first = { command: 'first', description: 'd', requestId: 'r1', sessionId: 's1' }
    const second = { ...first, command: 'second', requestId: 'r2' }
    setApprovalRequest(first)
    setApprovalRequest(second)
    setApprovalRequest(first)
    expect(
      sessionApprovalRequests('s1')
        .get()
        .map(request => request.requestId)
    ).toEqual(['r1', 'r2'])
    let finish!: () => void

    const gate = new Promise<void>(resolve => {
      finish = resolve
    })

    const replay = replayPendingApproval(
      fakeGateway(
        {
          'approval.pending': {
            approvals: [
              pendingApproval({ command: 'first', description: 'd', request_id: 'r1' }),
              pendingApproval({ command: 'second', description: 'd', request_id: 'r2' })
            ]
          },
          'approval.received': { acknowledged: true }
        },
        undefined,
        gate
      ).gateway,
      's1'
    )

    clearApprovalRequest('s1', 'r1')
    finish()
    await replay
    expect(sessionApprovalRequests('s1').get()).toEqual([second])
    clearAllPrompts('s1')
    expect(sessionApprovalRequests('s1').get()).toEqual([])
  })

  it('clears an absent approval without overwriting a newer live request', async () => {
    const old = { command: 'x', description: 'd', requestId: 'old', sessionId: 's1' }
    setApprovalRequest(old)
    await replayPendingApproval(fakeGateway({ 'approval.pending': { approvals: [] } }).gateway, 's1')
    expect($approvalRequest.get()).toBeNull()

    setApprovalRequest(old)
    let release!: () => void

    const gate = new Promise<void>(done => {
      release = done
    })

    const pending = replayPendingApproval(
      fakeGateway({ 'approval.pending': { approvals: [] } }, undefined, gate).gateway,
      's1'
    )

    setApprovalRequest({ ...old, requestId: 'new' })
    release()
    await pending
    expect(
      sessionApprovalRequests('s1')
        .get()
        .map(request => request.requestId)
    ).toEqual(['old', 'new'])
    clearApprovalRequest('s1', 'old')
    expect($approvalRequest.get()?.requestId).toBe('new')
  })

  it('does not replay a pending approval after the runtime is rejected as gone', async () => {
    const { calls, gateway } = fakeGateway({}, sessionNotFound)

    await replayPendingApproval(gateway, 'dead-runtime')
    await replayPendingApproval(gateway, 'dead-runtime')

    expect(calls).toHaveLength(1)
    expect(isSessionGone('dead-runtime')).toBe(true)
    expect($approvalRequest.get()).toBeNull()
  })

  it('propagates transient approval replay failures without latching the runtime', async () => {
    const { gateway } = fakeGateway({}, () => new Error('gateway timed out'))

    await expect(replayPendingApproval(gateway, 'transient-runtime')).rejects.toThrow('gateway timed out')
    expect(isSessionGone('transient-runtime')).toBe(false)
  })

  it('keeps approval receipt failures contained and marks the runtime gone', async () => {
    const { gateway } = fakeGateway({}, sessionNotFound)

    $activeSessionId.set('dead-runtime')

    await expect(
      receiveApprovalRequest(gateway, { command: 'x', description: 'd', requestId: 'r1', sessionId: 'dead-runtime' })
    ).resolves.toBeUndefined()

    expect(isSessionGone('dead-runtime')).toBe(true)
    expect($approvalRequest.get()?.requestId).toBe('r1')
  })

  it('propagates transient approval receipt failures without latching the runtime', async () => {
    const { gateway } = fakeGateway({}, () => new Error('gateway timed out'))

    setActiveSessionId('transient-runtime')

    await expect(
      receiveApprovalRequest(gateway, { command: 'x', description: 'd', requestId: 'r2', sessionId: 'transient-runtime' })
    ).rejects.toThrow('gateway timed out')

    expect(isSessionGone('transient-runtime')).toBe(false)
    expect($approvalRequest.get()?.requestId).toBe('r2')
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

    setClarifyRequest({
      choices: null,
      kind: 'single',
      multi_select: false,
      question: 'q',
      requestId: 'c1',
      sessionId: 's1'
    })
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
    const { calls, gateway } = fakeGateway({}, () => new Error('4001: session not found'))

    for (let i = 0; i < 5; i++) {
      await replayPendingApproval(gateway, 'dead-1')
    }

    expect(calls.map(([method]) => method)).toEqual(['approval.pending'])
  })

  it('keeps polling every other runtime', async () => {
    const { calls, gateway } = fakeGateway({ 'approval.pending': { approvals: [] } }, (_method, params) =>
      'session_id' in params && params.session_id === 'dead-1' ? new Error('4001: session not found') : undefined
    )

    await replayPendingApproval(gateway, 'dead-1')
    await replayPendingApproval(gateway, 'dead-1')
    await replayPendingApproval(gateway, 'alive-1')
    await replayPendingApproval(gateway, 'alive-1')

    expect(calls.map(([, params]) => ('session_id' in params ? params.session_id : null))).toEqual([
      'dead-1',
      'alive-1',
      'alive-1'
    ])
  })

  it('does not latch on a transient failure', async () => {
    const { calls, gateway } = fakeGateway({}, () => new Error('websocket disconnected'))

    await replayPendingApproval(gateway, 's1').catch(() => undefined)
    await replayPendingApproval(gateway, 's1').catch(() => undefined)

    expect(calls).toHaveLength(2)
  })

  it('polls again once the gone-latch is cleared', async () => {
    let dead = true

    const { calls, gateway } = fakeGateway({ 'approval.pending': { approvals: [] } }, () =>
      dead ? new Error('4001: session not found') : undefined
    )

    await replayPendingApproval(gateway, 's1')
    await replayPendingApproval(gateway, 's1')
    expect(calls).toHaveLength(1)

    dead = false
    resetBackgroundPollingGuard()
    await replayPendingApproval(gateway, 's1')

    expect(calls).toHaveLength(2)
  })
})
