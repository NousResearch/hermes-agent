import { JsonRpcGatewayError } from '@hermes/shared'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const { refreshLegacyGoal } = vi.hoisted(() => ({ refreshLegacyGoal: vi.fn() }))

vi.mock('./goals', async importOriginal => ({
  ...(await importOriginal()),
  refreshSessionGoal: refreshLegacyGoal
}))

import { $gateway } from './gateway'
import { resetBackgroundPollingGuard } from './runtime-gone'
import {
  $sessionControlBySession,
  applySessionControlSnapshot,
  applySessionControlUpdate,
  clearAllSessionControl,
  clearSessionControl,
  parseSessionControlSnapshot,
  refreshSessionControl,
  refreshSupportedSessionControlAfterTurn,
  runSessionControlAction,
  type SessionControlSnapshot
} from './session-control'

const FULL_SNAPSHOT: SessionControlSnapshot = {
  goal: {
    contract: {
      boundaries: 'desktop store only',
      constraints: 'do not lose state',
      outcome: 'session control is hydrated',
      stop_when: 'a human decision is required',
      verification: 'focused tests pass'
    },
    created_at: 1_700_000_000,
    gates: [{ attempts: 0, command: 'npm test', last_exit_code: null, max_retries: 2, timeout_seconds: 60 }],
    max_turns: 20,
    status: 'active',
    subgoals: ['write tests', 'repair state'],
    title: 'Repair session control',
    turns_used: 3,
    updated_at: 1_700_000_100,
    wait_barrier: { reason: 'waiting for deploy', type: 'until', until_at: 1_700_000_200 }
  },
  heartbeat: {
    created_at: 1_700_000_000,
    fire_count: 5,
    interval_seconds: 600,
    last_fired_at: 1_700_000_100,
    prompt: 'check health',
    status: 'active'
  },
  loop: {
    awaiting_response: false,
    created_at: 1_700_000_000,
    current_delay: 300,
    deferred_by_goal: false,
    interval_seconds: 300,
    last_fired_at: 1_700_000_100,
    max_ticks: 10,
    mode: 'interval',
    next_due_at: 1_700_000_400,
    prompt: 'check build status',
    status: 'active',
    ticks_fired: 3,
    times: 3,
    until: ''
  },
  revision: 'revision-1',
  updated_at: 1_700_000_100
}

function deferred<T>() {
  let resolve!: (value: T) => void
  let reject!: (error: unknown) => void

  const promise = new Promise<T>((resolvePromise, rejectPromise) => {
    resolve = resolvePromise
    reject = rejectPromise
  })

  return { promise, reject, resolve }
}

function useGateway(request: (method: string, params: Record<string, unknown>) => Promise<unknown>): void {
  $gateway.set({ request } as never)
}

describe('session-control store', () => {
  beforeEach(() => {
    refreshLegacyGoal.mockReset()
    clearAllSessionControl()
  })

  afterEach(() => {
    $gateway.set(null as never)
    resetBackgroundPollingGuard()
    clearAllSessionControl()
  })

  it('parses the exact persisted goal, loop, heartbeat, and wait-barrier shapes into fresh data', () => {
    const parsed = parseSessionControlSnapshot(FULL_SNAPSHOT)

    expect(parsed).toEqual(FULL_SNAPSHOT)
    expect(parsed).not.toBe(FULL_SNAPSHOT)
    expect(parsed!.goal).not.toBe(FULL_SNAPSHOT.goal)
    expect(parsed!.goal!.contract).not.toBe(FULL_SNAPSHOT.goal!.contract)
    expect(parsed!.loop!.mode).toBe('interval')
    expect(parsed!.goal!.wait_barrier).toEqual({ reason: 'waiting for deploy', type: 'until', until_at: 1_700_000_200 })
  })

  it.each([
    ['unknown goal status', { ...FULL_SNAPSHOT, goal: { ...FULL_SNAPSHOT.goal!, status: 'waiting' } }],
    ['non-finite top-level timestamp', { ...FULL_SNAPSHOT, updated_at: Number.NaN }],
    ['malformed goal contract', { ...FULL_SNAPSHOT, goal: { ...FULL_SNAPSHOT.goal!, contract: { outcome: 3 } } }],
    [
      'gate output that is not in the allowlisted summary',
      {
        ...FULL_SNAPSHOT,
        goal: { ...FULL_SNAPSHOT.goal!, gates: [{ ...FULL_SNAPSHOT.goal!.gates[0], last_output_tail: 'leak' }] }
      }
    ],
    [
      'wait target that does not match its discriminator',
      { ...FULL_SNAPSHOT, goal: { ...FULL_SNAPSHOT.goal!, wait_barrier: { reason: 'pid', target: '7', type: 'pid' } } }
    ],
    ['unknown loop mode', { ...FULL_SNAPSHOT, loop: { ...FULL_SNAPSHOT.loop!, mode: 'fixed' } }],
    ['malformed heartbeat', { ...FULL_SNAPSHOT, heartbeat: { ...FULL_SNAPSHOT.heartbeat!, fire_count: 'five' } }],
    ['unknown top-level field', { ...FULL_SNAPSHOT, unsupported: true }]
  ])('rejects %s without accepting a partial snapshot', (_name, value) => {
    expect(parseSessionControlSnapshot(value)).toBeNull()
  })

  it('falls back once on the unsupported transition and suppresses future compatibility retries', async () => {
    const request = vi.fn(async () => {
      throw new JsonRpcGatewayError('method not found', { code: -32601 })
    })

    useGateway(request)

    await refreshSessionControl('s1')
    await refreshSessionControl('s1')

    expect($sessionControlBySession.get().s1).toMatchObject({
      capability: 'unsupported',
      loading: false,
      snapshot: null
    })
    expect(refreshLegacyGoal).toHaveBeenCalledTimes(1)
    expect(refreshLegacyGoal).toHaveBeenCalledWith('s1')
    expect(request).toHaveBeenCalledTimes(1)
  })

  it('retains the last good snapshot and reports a bounded ordinary read error', async () => {
    applySessionControlSnapshot('s1', FULL_SNAPSHOT)
    const message = 'x'.repeat(500)
    useGateway(
      vi.fn(async () => {
        throw new Error(message)
      })
    )

    await expect(refreshSessionControl('s1', { background: true })).resolves.toBeDefined()

    const entry = $sessionControlBySession.get().s1!
    expect(entry.snapshot!.revision).toBe(FULL_SNAPSHOT.revision)
    expect(entry.capability).toBe('supported')
    expect(entry.error).toHaveLength(240)
    expect(entry.loading).toBe(false)
  })

  it('downgrades a current action method-not-found once, preserves its snapshot, and still rejects', async () => {
    applySessionControlSnapshot('s1', FULL_SNAPSHOT)
    useGateway(
      vi.fn(async () => {
        throw new JsonRpcGatewayError('method not found', { code: -32601 })
      })
    )

    await expect(runSessionControlAction('s1', 'goal.pause')).rejects.toThrow('method not found')

    expect($sessionControlBySession.get().s1).toMatchObject({
      capability: 'unsupported',
      error: null,
      loading: false,
      pendingAction: null,
      snapshot: { revision: FULL_SNAPSHOT.revision }
    })
    expect(refreshLegacyGoal).toHaveBeenCalledTimes(1)
    expect(refreshLegacyGoal).toHaveBeenCalledWith('s1')
  })

  it('does not let a method-not-found action response downgrade a newer event snapshot', async () => {
    applySessionControlSnapshot('s1', FULL_SNAPSHOT)
    const response = deferred<unknown>()
    useGateway(vi.fn(() => response.promise))

    const action = runSessionControlAction('s1', 'goal.pause')
    applySessionControlUpdate('s1', { ...FULL_SNAPSHOT, revision: 'event-newer' })
    response.reject(new JsonRpcGatewayError('method not found', { code: -32601 }))

    await expect(action).rejects.toThrow('method not found')
    expect($sessionControlBySession.get().s1).toMatchObject({
      capability: 'supported',
      error: 'method not found',
      loading: false,
      pendingAction: null,
      snapshot: { revision: 'event-newer' }
    })
    expect(refreshLegacyGoal).not.toHaveBeenCalled()
  })

  it('does not let a stale method-not-found response disturb a newer action', async () => {
    applySessionControlSnapshot('s1', FULL_SNAPSHOT)
    const olderResponse = deferred<unknown>()
    const newerResponse = deferred<unknown>()
    useGateway(
      vi
        .fn()
        .mockImplementationOnce(() => olderResponse.promise)
        .mockImplementationOnce(() => newerResponse.promise)
    )

    const olderAction = runSessionControlAction('s1', 'goal.pause')
    const newerAction = runSessionControlAction('s1', 'goal.resume')
    olderResponse.reject(new JsonRpcGatewayError('method not found', { code: -32601 }))

    await expect(olderAction).rejects.toThrow('method not found')
    expect($sessionControlBySession.get().s1).toMatchObject({
      capability: 'supported',
      error: null,
      loading: true,
      pendingAction: 'goal.resume',
      snapshot: { revision: FULL_SNAPSHOT.revision }
    })
    expect(refreshLegacyGoal).not.toHaveBeenCalled()

    newerResponse.resolve({
      control: { ...FULL_SNAPSHOT, revision: 'newer-action' },
      dispatch: { display: null, message: null, notice: null, output: 'resumed', type: 'exec' }
    })
    await expect(newerAction).resolves.toMatchObject({ output: 'resumed' })
    expect($sessionControlBySession.get().s1).toMatchObject({
      loading: false,
      pendingAction: null,
      snapshot: { revision: 'newer-action' }
    })
  })

  it('submits only the requested action and exposes its pending state on that session', async () => {
    const response = deferred<unknown>()
    const request = vi.fn(() => response.promise)
    useGateway(request)

    const action = runSessionControlAction('s1', 'subgoal.add', { text: 'verify hydration' })
    expect($sessionControlBySession.get().s1).toMatchObject({
      capability: 'unknown',
      loading: true,
      pendingAction: 'subgoal.add'
    })
    expect($sessionControlBySession.get().s2).toBeUndefined()
    expect(request).toHaveBeenCalledWith('session.control', {
      action: 'subgoal.add',
      args: { text: 'verify hydration' },
      session_id: 's1'
    })

    response.resolve({
      control: { ...FULL_SNAPSHOT, revision: 'action-revision' },
      dispatch: { display: null, message: null, notice: null, output: 'added', type: 'exec' }
    })

    await expect(action).resolves.toEqual({ display: null, message: null, notice: null, output: 'added', type: 'exec' })
    expect($sessionControlBySession.get().s1).toMatchObject({
      capability: 'supported',
      error: null,
      pendingAction: null
    })
  })

  it('rejects a failed action truthfully while retaining the good snapshot and clearing its pending state', async () => {
    applySessionControlSnapshot('s1', FULL_SNAPSHOT)
    useGateway(
      vi.fn(async () => {
        throw new Error('backend unavailable')
      })
    )

    await expect(runSessionControlAction('s1', 'goal.pause')).rejects.toThrow('backend unavailable')

    expect($sessionControlBySession.get().s1).toMatchObject({ error: 'backend unavailable', pendingAction: null })
    expect($sessionControlBySession.get().s1!.snapshot!.revision).toBe(FULL_SNAPSHOT.revision)
  })

  it('keeps a pending action current through an event and publishes its later failure', async () => {
    applySessionControlSnapshot('s1', FULL_SNAPSHOT)
    const response = deferred<unknown>()
    useGateway(vi.fn(() => response.promise))

    const action = runSessionControlAction('s1', 'goal.pause')
    expect($sessionControlBySession.get().s1).toMatchObject({ loading: true, pendingAction: 'goal.pause' })

    applySessionControlUpdate('s1', { ...FULL_SNAPSHOT, revision: 'event-newer' })
    expect($sessionControlBySession.get().s1).toMatchObject({
      error: null,
      loading: true,
      pendingAction: 'goal.pause',
      snapshot: { revision: 'event-newer' }
    })

    response.reject(new Error('backend unavailable'))
    await expect(action).rejects.toThrow('backend unavailable')

    expect($sessionControlBySession.get().s1).toMatchObject({
      error: 'backend unavailable',
      loading: false,
      pendingAction: null,
      snapshot: { revision: 'event-newer' }
    })
  })

  it('does not let a late read overwrite a newer event', async () => {
    const slow = deferred<unknown>()
    useGateway(vi.fn(() => slow.promise))

    const read = refreshSessionControl('s1')
    expect($sessionControlBySession.get().s1).toMatchObject({ loading: true, pendingAction: null })

    applySessionControlUpdate('s1', { ...FULL_SNAPSHOT, revision: 'event-newer' })
    expect($sessionControlBySession.get().s1).toMatchObject({
      loading: false,
      pendingAction: null,
      snapshot: { revision: 'event-newer' }
    })

    slow.resolve({ control: { ...FULL_SNAPSHOT, revision: 'read-stale' } })
    await read

    expect($sessionControlBySession.get().s1).toMatchObject({
      loading: false,
      pendingAction: null,
      snapshot: { revision: 'event-newer' }
    })
  })

  it('does not let a late read overwrite a newer action response', async () => {
    const slow = deferred<unknown>()

    const request = vi
      .fn()
      .mockImplementationOnce(() => slow.promise)
      .mockResolvedValueOnce({
        control: { ...FULL_SNAPSHOT, revision: 'action-newer' },
        dispatch: { display: null, message: null, notice: null, output: 'paused', type: 'exec' }
      })

    useGateway(request)

    const read = refreshSessionControl('s1')
    await runSessionControlAction('s1', 'goal.pause')
    slow.resolve({ control: { ...FULL_SNAPSHOT, revision: 'read-stale' } })
    await read

    expect($sessionControlBySession.get().s1!.snapshot!.revision).toBe('action-newer')
  })

  it('a gateway-switch wipe drops every entry and strands in-flight responses from the old backend', async () => {
    applySessionControlSnapshot('supported', FULL_SNAPSHOT)
    const slowRead = deferred<unknown>()
    const slowAction = deferred<unknown>()
    useGateway(vi.fn((method: string) => (method === 'session.control' ? slowAction.promise : slowRead.promise)))

    const read = refreshSessionControl('probing')
    const action = runSessionControlAction('supported', 'goal.pause')

    clearAllSessionControl()
    expect($sessionControlBySession.get()).toEqual({})

    slowRead.resolve({ control: FULL_SNAPSHOT })
    slowAction.resolve({
      control: { ...FULL_SNAPSHOT, revision: 'stale-action' },
      dispatch: { display: null, message: null, notice: null, output: 'paused', type: 'exec' }
    })
    await read
    await action

    expect($sessionControlBySession.get()).toEqual({})
  })

  it('does not let a late read repopulate a cleared session', async () => {
    const slow = deferred<unknown>()
    useGateway(vi.fn(() => slow.promise))

    const read = refreshSessionControl('s1')
    clearSessionControl('s1')
    slow.resolve({ control: FULL_SNAPSHOT })
    await read

    expect($sessionControlBySession.get().s1).toBeUndefined()
  })

  it('does not let an older read overwrite a newer read', async () => {
    const slow = deferred<unknown>()
    const fast = deferred<unknown>()

    const request = vi
      .fn()
      .mockImplementationOnce(() => slow.promise)
      .mockImplementationOnce(() => fast.promise)

    useGateway(request)

    const older = refreshSessionControl('s1')
    const newer = refreshSessionControl('s1')
    fast.resolve({ control: { ...FULL_SNAPSHOT, revision: 'newer-read' } })
    await newer
    slow.resolve({ control: { ...FULL_SNAPSHOT, revision: 'older-read' } })
    await older

    expect($sessionControlBySession.get().s1!.snapshot!.revision).toBe('newer-read')
  })

  it('does not let a late post-turn refresh overwrite a newer event', async () => {
    applySessionControlSnapshot('s1', FULL_SNAPSHOT)
    const slow = deferred<unknown>()
    useGateway(vi.fn(() => slow.promise))

    const refresh = refreshSupportedSessionControlAfterTurn('s1')
    applySessionControlUpdate('s1', { ...FULL_SNAPSHOT, revision: 'event-newer' })
    expect($sessionControlBySession.get().s1).toMatchObject({
      loading: false,
      pendingAction: null,
      snapshot: { revision: 'event-newer' }
    })

    slow.resolve({ control: { ...FULL_SNAPSHOT, revision: 'post-turn-stale' } })
    await refresh

    expect($sessionControlBySession.get().s1).toMatchObject({
      loading: false,
      pendingAction: null,
      snapshot: { revision: 'event-newer' }
    })
  })

  it('applies an authoritative action response after an intermediate event', async () => {
    const slow = deferred<unknown>()
    useGateway(vi.fn(() => slow.promise))

    const action = runSessionControlAction('s1', 'goal.pause')
    expect($sessionControlBySession.get().s1).toMatchObject({ loading: true, pendingAction: 'goal.pause' })

    applySessionControlUpdate('s1', { ...FULL_SNAPSHOT, revision: 'event-newer' })
    expect($sessionControlBySession.get().s1).toMatchObject({
      loading: true,
      pendingAction: 'goal.pause',
      snapshot: { revision: 'event-newer' }
    })

    slow.resolve({
      control: { ...FULL_SNAPSHOT, revision: 'action-stale' },
      dispatch: { display: null, message: null, notice: null, output: 'paused', type: 'exec' }
    })

    await expect(action).resolves.toMatchObject({ output: 'paused', type: 'exec' })
    expect($sessionControlBySession.get().s1).toMatchObject({
      loading: false,
      pendingAction: null,
      snapshot: { revision: 'action-stale' }
    })
  })
})
