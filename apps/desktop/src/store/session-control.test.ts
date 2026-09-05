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
  clearSessionControl,
  parseSessionControlSnapshot,
  refreshSessionControl,
  refreshSupportedSessionControlAfterTurn,
  resetSessionControlAfterGatewayRebind,
  resetSessionControlForTests,
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
    resetSessionControlForTests()
  })

  afterEach(() => {
    $gateway.set(null as never)
    resetBackgroundPollingGuard()
    resetSessionControlForTests()
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

  it('preserves entry and snapshot identity for a same-revision snapshot with no other state change', () => {
    applySessionControlSnapshot('s1', FULL_SNAPSHOT)
    const first = $sessionControlBySession.get().s1

    applySessionControlSnapshot('s1', { ...FULL_SNAPSHOT })
    const second = $sessionControlBySession.get().s1

    expect(second).toBe(first)
    expect(second!.snapshot).toBe(first!.snapshot)
  })

  it('creates foreground loading state and leaves background hydration visually quiet', async () => {
    const first = deferred<unknown>()
    const request = vi.fn(() => first.promise)
    useGateway(request)

    const foreground = refreshSessionControl('s1')
    expect($sessionControlBySession.get().s1).toMatchObject({
      capability: 'unknown',
      error: null,
      loading: true,
      pendingAction: null
    })
    first.resolve({ control: FULL_SNAPSHOT })
    await foreground

    const second = deferred<unknown>()
    request.mockImplementationOnce(() => second.promise)
    const background = refreshSessionControl('s1', { background: true })

    expect($sessionControlBySession.get().s1).toMatchObject({ capability: 'supported', loading: false })
    second.resolve({ control: { ...FULL_SNAPSHOT, revision: 'background-revision' } })
    await background
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

  it('lets a precise gateway-rebind seam make an unsupported session probeable again', async () => {
    useGateway(
      vi.fn(async () => {
        throw new JsonRpcGatewayError('method not found', { code: -32601 })
      })
    )
    await refreshSessionControl('s1')

    resetSessionControlAfterGatewayRebind()
    expect($sessionControlBySession.get().s1).toMatchObject({ capability: 'unknown', loading: false })

    const request = vi.fn(async () => ({ control: FULL_SNAPSHOT }))
    useGateway(request)
    await refreshSessionControl('s1')
    expect(request).toHaveBeenCalledTimes(1)
    expect($sessionControlBySession.get().s1!.capability).toBe('supported')
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

  it('marks a session gone without publishing an ordinary hydration error', async () => {
    useGateway(
      vi.fn(async () => {
        throw new JsonRpcGatewayError('session not found', { code: 4001 })
      })
    )

    await refreshSessionControl('gone')
    expect($sessionControlBySession.get().gone).toMatchObject({ error: null, loading: false })
  })

  it('rejects a session-gone action truthfully while clearing only its pending state', async () => {
    applySessionControlSnapshot('s1', FULL_SNAPSHOT)
    useGateway(
      vi.fn(async () => {
        throw new JsonRpcGatewayError('session not found', { code: 4001 })
      })
    )

    await expect(runSessionControlAction('s1', 'goal.pause')).rejects.toThrow('session not found')
    expect($sessionControlBySession.get().s1).toMatchObject({ error: null, pendingAction: null })
    expect($sessionControlBySession.get().s1!.snapshot!.revision).toBe(FULL_SNAPSHOT.revision)
  })

  it('makes an event-first session supported and leaves malformed events as a no-op', () => {
    applySessionControlUpdate('s1', FULL_SNAPSHOT)
    const first = $sessionControlBySession.get().s1

    applySessionControlUpdate('s1', { ...FULL_SNAPSHOT, loop: { ...FULL_SNAPSHOT.loop!, mode: 'fixed' } })

    expect(first).toMatchObject({ capability: 'supported', pendingAction: null })
    expect($sessionControlBySession.get().s1).toBe(first)
  })

  it('submits only the requested action and exposes its pending state on that session', async () => {
    const response = deferred<unknown>()
    const request = vi.fn(() => response.promise)
    useGateway(request)

    const action = runSessionControlAction('s1', 'subgoal.add', { text: 'verify hydration' })
    expect($sessionControlBySession.get().s1).toMatchObject({
      capability: 'unknown',
      loading: false,
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

  it('rejects malformed action data without promoting an unknown capability', async () => {
    useGateway(
      vi.fn(async () => ({
        control: FULL_SNAPSHOT,
        dispatch: { display: null, message: null, notice: null, output: null, type: 'unknown' }
      }))
    )

    await expect(runSessionControlAction('s1', 'goal.pause')).rejects.toThrow('Invalid session.control action response')
    expect($sessionControlBySession.get().s1).toMatchObject({
      capability: 'unknown',
      pendingAction: null,
      snapshot: null
    })
  })

  it('does not let a late read overwrite a newer event', async () => {
    const slow = deferred<unknown>()
    useGateway(vi.fn(() => slow.promise))

    const read = refreshSessionControl('s1')
    applySessionControlUpdate('s1', { ...FULL_SNAPSHOT, revision: 'event-newer' })
    slow.resolve({ control: { ...FULL_SNAPSHOT, revision: 'read-stale' } })
    await read

    expect($sessionControlBySession.get().s1!.snapshot!.revision).toBe('event-newer')
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
    slow.resolve({ control: { ...FULL_SNAPSHOT, revision: 'post-turn-stale' } })
    await refresh

    expect($sessionControlBySession.get().s1!.snapshot!.revision).toBe('event-newer')
  })

  it('skips post-turn refreshes until the session is known to support the control RPC', async () => {
    const request = vi.fn(async () => ({ control: FULL_SNAPSHOT }))
    useGateway(request)

    await refreshSupportedSessionControlAfterTurn('unknown')
    expect(request).not.toHaveBeenCalled()

    applySessionControlUpdate('unsupported', FULL_SNAPSHOT)
    useGateway(
      vi.fn(async () => {
        throw new JsonRpcGatewayError('method not found', { code: -32601 })
      })
    )
    await refreshSessionControl('unsupported')

    const unsupportedRequest = vi.fn(async () => ({ control: FULL_SNAPSHOT }))
    useGateway(unsupportedRequest)
    await refreshSupportedSessionControlAfterTurn('unsupported')
    expect(unsupportedRequest).not.toHaveBeenCalled()
  })

  it('returns a valid stale action dispatch without letting it overwrite a newer event', async () => {
    const slow = deferred<unknown>()
    useGateway(vi.fn(() => slow.promise))

    const action = runSessionControlAction('s1', 'goal.pause')
    applySessionControlUpdate('s1', { ...FULL_SNAPSHOT, revision: 'event-newer' })
    slow.resolve({
      control: { ...FULL_SNAPSHOT, revision: 'action-stale' },
      dispatch: { display: null, message: null, notice: null, output: 'paused', type: 'exec' }
    })

    await expect(action).resolves.toMatchObject({ output: 'paused', type: 'exec' })
    expect($sessionControlBySession.get().s1!.snapshot!.revision).toBe('event-newer')
  })

  it('does not schedule a timer or poller while hydrating or dispatching control state', async () => {
    const setIntervalSpy = vi.spyOn(globalThis, 'setInterval')
    const setTimeoutSpy = vi.spyOn(globalThis, 'setTimeout')
    useGateway(
      vi.fn(async method =>
        method === 'session.control.read'
          ? { control: FULL_SNAPSHOT }
          : {
              control: FULL_SNAPSHOT,
              dispatch: { display: null, message: null, notice: null, output: null, type: 'exec' }
            }
      )
    )

    await refreshSessionControl('s1')
    await runSessionControlAction('s1', 'goal.pause')
    await refreshSupportedSessionControlAfterTurn('s1')

    expect(setIntervalSpy).not.toHaveBeenCalled()
    expect(setTimeoutSpy).not.toHaveBeenCalled()
  })
})
