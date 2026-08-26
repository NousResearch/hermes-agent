import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { JsonRpcGatewayError } from '@hermes/shared'

import {
  $backgroundStatusBySession,
  dismissBackgroundProcess,
  isSessionGoneForBackgroundPolling,
  reconcileBackgroundProcesses,
  refreshBackgroundProcesses,
  resetBackgroundPollingGuard
} from './composer-status'
import { $gateway } from './gateway'

const SID = 'sess-1'

const running = (id: string, command = `cmd ${id}`) => ({ command, session_id: id, status: 'running' })

const exited = (id: string, exit_code = 0, command = `cmd ${id}`) => ({
  command,
  exit_code,
  session_id: id,
  status: 'exited'
})

const items = () => $backgroundStatusBySession.get()[SID] ?? []

describe('reconcileBackgroundProcesses', () => {
  beforeEach(() => {
    // Fake timers so the success self-clear (a real setTimeout) is deterministic
    // and never leaks a pending timer between tests.
    vi.useFakeTimers()
    $backgroundStatusBySession.set({})
  })

  afterEach(() => {
    vi.clearAllTimers()
    vi.useRealTimers()
  })

  it('maps registry entries to status items', () => {
    reconcileBackgroundProcesses(SID, [running('a'), exited('b', 0), exited('c', 1)])

    expect(items().map(i => [i.id, i.state])).toEqual([
      ['a', 'running'],
      ['b', 'done'],
      ['c', 'failed']
    ])
    expect(items()[2]!.exitCode).toBe(1)
  })

  it('keeps row order stable when a process flips state or the snapshot reorders', () => {
    reconcileBackgroundProcesses(SID, [running('a'), running('b')])
    // Snapshot arrives reordered AND `a` has exited — rows must not move.
    reconcileBackgroundProcesses(SID, [running('b'), exited('a', 0)])

    expect(items().map(i => [i.id, i.state])).toEqual([
      ['a', 'done'],
      ['b', 'running']
    ])
  })

  it('appends new processes after existing rows', () => {
    reconcileBackgroundProcesses(SID, [running('a')])
    reconcileBackgroundProcesses(SID, [running('b'), running('a')])

    expect(items().map(i => i.id)).toEqual(['a', 'b'])
  })

  it('preserves object identity for unchanged rows (memo stability)', () => {
    reconcileBackgroundProcesses(SID, [running('a'), running('b')])
    const [a1] = items()

    reconcileBackgroundProcesses(SID, [running('a'), exited('b', 0)])
    const [a2, b2] = items()

    expect(a2).toBe(a1)
    expect(b2!.state).toBe('done')
  })

  it('is a no-op store write when nothing changed', () => {
    reconcileBackgroundProcesses(SID, [running('a')])
    const before = $backgroundStatusBySession.get()

    reconcileBackgroundProcesses(SID, [running('a')])

    expect($backgroundStatusBySession.get()).toBe(before)
  })

  it('never resurrects a dismissed process while the registry still reports it', () => {
    reconcileBackgroundProcesses(SID, [exited('a', 0), running('b')])
    dismissBackgroundProcess(SID, 'a')

    reconcileBackgroundProcesses(SID, [exited('a', 0), running('b')])

    expect(items().map(i => i.id)).toEqual(['b'])
  })

  it('forgets a dismissal once the registry prunes the process', () => {
    reconcileBackgroundProcesses(SID, [exited('a', 0)])
    dismissBackgroundProcess(SID, 'a')

    // Registry pruned it…
    reconcileBackgroundProcesses(SID, [])
    // …so a future process reusing the id (new spawn) shows again.
    reconcileBackgroundProcesses(SID, [running('a')])

    expect(items().map(i => i.id)).toEqual(['a'])
  })

  it('drops the session key entirely when the last row goes away', () => {
    reconcileBackgroundProcesses(SID, [running('a')])
    reconcileBackgroundProcesses(SID, [])

    expect($backgroundStatusBySession.get()).toEqual({})
  })

  // The self-clear path calls dismissBackgroundProcess, which records the id in
  // the module-level dismissed set; use a fresh session per test so that record
  // can't bleed into another test's reconcile.
  const itemsOf = (sid: string) => $backgroundStatusBySession.get()[sid] ?? []

  it('self-clears a finished success after a short linger', () => {
    reconcileBackgroundProcesses('sess-clear', [exited('a', 0)])
    expect(itemsOf('sess-clear').map(i => i.id)).toEqual(['a'])

    vi.advanceTimersByTime(5_000)

    expect(itemsOf('sess-clear')).toEqual([])
  })

  it('self-clears a failed task too, but only after a longer linger', () => {
    reconcileBackgroundProcesses('sess-fail', [exited('a', 1)])

    // Still visible after the success window — the failure gets a longer one so
    // its exit code stays readable.
    vi.advanceTimersByTime(5_000)
    expect(itemsOf('sess-fail').map(i => [i.id, i.state])).toEqual([['a', 'failed']])

    vi.advanceTimersByTime(10_000)
    expect(itemsOf('sess-fail')).toEqual([])
  })

  it('never self-clears a still-running task', () => {
    reconcileBackgroundProcesses('sess-run', [running('a')])

    vi.advanceTimersByTime(60_000)

    expect(itemsOf('sess-run').map(i => i.id)).toEqual(['a'])
  })

  it('arms the self-clear only once a task finishes', () => {
    reconcileBackgroundProcesses('sess-arm', [running('a')])
    vi.advanceTimersByTime(60_000)
    // Still running after a minute — nothing scheduled yet.
    expect(itemsOf('sess-arm').map(i => i.id)).toEqual(['a'])

    reconcileBackgroundProcesses('sess-arm', [exited('a', 0)])
    vi.advanceTimersByTime(5_000)

    expect(itemsOf('sess-arm')).toEqual([])
  })
})

// ── Dead-session polling guard (#94219 fallout) ──────────────────────────────
// The status stack polls `process.list` every 5s while a background row is on
// screen. `process.list` is session-scoped: against a runtime id the gateway no
// longer holds it returns 4001 "session not found". That failure was swallowed
// as "transient socket loss", so the poll re-sent the SAME dead id every 5s for
// the life of the window — 18,614 rejections against a single runtime id in one
// day on a real machine, and the log line the user reads as "session not found".
//
// A gone session is terminal, not transient: stop polling it.
describe('refreshBackgroundProcesses dead-session guard', () => {
  beforeEach(() => {
    $backgroundStatusBySession.set({})
    resetBackgroundPollingGuard()
  })

  afterEach(() => {
    $gateway.set(null as never)
    resetBackgroundPollingGuard()
  })

  it('classifies a 4001 session-not-found as gone, and a timeout as transient', () => {
    expect(isSessionGoneForBackgroundPolling(new Error('session not found'))).toBe(true)
    expect(isSessionGoneForBackgroundPolling(new Error('4001 session not found'))).toBe(true)
    expect(isSessionGoneForBackgroundPolling(new Error('Session Not Found'))).toBe(true)

    // Transient failures must NOT latch the guard — the session may still be alive.
    expect(isSessionGoneForBackgroundPolling(new Error('request timed out after 30s: process.list'))).toBe(false)
    expect(isSessionGoneForBackgroundPolling(new Error('not connected'))).toBe(false)
  })

  it('stops re-polling a session after the gateway reports it gone', async () => {
    const request = vi.fn(async () => {
      throw new Error('session not found')
    })

    $gateway.set({ request } as never)

    // First poll discovers the session is gone.
    await refreshBackgroundProcesses(SID)
    expect(request).toHaveBeenCalledTimes(1)

    // Every subsequent tick must be suppressed. Before the fix these all went
    // to the wire and each produced another gateway-side 4001.
    await refreshBackgroundProcesses(SID)
    await refreshBackgroundProcesses(SID)
    await refreshBackgroundProcesses(SID)

    expect(request).toHaveBeenCalledTimes(1)
  })

  it('keeps polling after a transient failure', async () => {
    const request = vi.fn(async () => {
      throw new Error('request timed out after 30s: process.list')
    })

    $gateway.set({ request } as never)

    await refreshBackgroundProcesses(SID)
    await refreshBackgroundProcesses(SID)

    // A timeout is not proof of death — the poll must retry.
    expect(request).toHaveBeenCalledTimes(2)
  })

  it('does not suppress a different, healthy session', async () => {
    const request = vi.fn(async (_method: string, params?: Record<string, unknown>) => {
      if (params?.session_id === SID) {
        throw new Error('session not found')
      }

      return { processes: [] }
    })

    $gateway.set({ request } as never)

    await refreshBackgroundProcesses(SID)
    await refreshBackgroundProcesses(SID)
    await refreshBackgroundProcesses('sess-healthy')
    await refreshBackgroundProcesses('sess-healthy')

    const targets = request.mock.calls.map(c => (c[1] as { session_id?: string } | undefined)?.session_id)
    expect(targets.filter(t => t === SID)).toHaveLength(1)
    expect(targets.filter(t => t === 'sess-healthy')).toHaveLength(2)
  })

  it('resumes polling a session that comes back (guard cleared on rebind)', async () => {
    const request = vi.fn(async () => {
      throw new Error('session not found')
    })

    $gateway.set({ request } as never)

    await refreshBackgroundProcesses(SID)
    await refreshBackgroundProcesses(SID)
    expect(request).toHaveBeenCalledTimes(1)

    // A fresh runtime bound to this session id clears the guard.
    resetBackgroundPollingGuard(SID)
    await refreshBackgroundProcesses(SID)

    expect(request).toHaveBeenCalledTimes(2)
  })
})

// ── Review-thread hardenings on the guard (#94950) ───────────────────────────
describe('refreshBackgroundProcesses dead-session guard hardenings', () => {
  beforeEach(() => {
    $backgroundStatusBySession.set({})
    resetBackgroundPollingGuard()
  })

  afterEach(() => {
    $gateway.set(null as never)
    resetBackgroundPollingGuard()
  })

  it('matches the structured 4001 code, not a message substring, when a code is present', () => {
    // Structured gateway rejection: the code decides, both directions.
    expect(isSessionGoneForBackgroundPolling(new JsonRpcGatewayError('session not found', { code: 4001 }))).toBe(true)
    expect(isSessionGoneForBackgroundPolling(new JsonRpcGatewayError('gone', { code: 4001 }))).toBe(true)

    // An unrelated coded error whose text merely MENTIONS the phrase must not
    // latch — that would freeze the status stack on a healthy session.
    expect(
      isSessionGoneForBackgroundPolling(
        new JsonRpcGatewayError('tool failed: upstream said session not found', { code: 5007 })
      )
    ).toBe(false)

    // Codeless errors keep the message fallback (legacy frames).
    expect(isSessionGoneForBackgroundPolling(new JsonRpcGatewayError('session not found'))).toBe(true)
  })

  it('a full guard reset (runtime re-mint) resumes polling every latched session', async () => {
    const request = vi.fn(async () => {
      throw new JsonRpcGatewayError('session not found', { code: 4001 })
    })

    $gateway.set({ request } as never)

    await refreshBackgroundProcesses(SID)
    await refreshBackgroundProcesses('sess-2')
    await refreshBackgroundProcesses(SID)
    await refreshBackgroundProcesses('sess-2')
    expect(request).toHaveBeenCalledTimes(2)

    // Gateway reconnect re-mints runtimes: the no-arg reset (wired at the
    // reconnect seams) must clear every latched id, not just one.
    resetBackgroundPollingGuard()
    await refreshBackgroundProcesses(SID)
    await refreshBackgroundProcesses('sess-2')

    expect(request).toHaveBeenCalledTimes(4)
  })
})
