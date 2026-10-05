import { EventEmitter } from 'node:events'

import { describe, expect, it, vi } from 'vitest'

import { killChildOnAbort } from './git-abort'

/**
 * A child-like object over a real EventEmitter, so `emit` has Node's exact
 * behaviour: an 'error' with no listener throws. That is the property the
 * escalation depends on — a spent once-listener is a crash, not a no-op.
 * `kill` records the signals; `removeListener` records what teardown dropped.
 */
function fakeChild() {
  const kills: Array<NodeJS.Signals | number | undefined> = []
  const removed: string[] = []
  const emitter = new EventEmitter()

  const child = {
    // Undefined until a test says otherwise: a real ChildProcess has a pid
    // only once the spawn has happened.
    pid: undefined as number | undefined,
    // null while the process runs, set once it exited or was signalled.
    exitCode: null as number | null,
    signalCode: null as NodeJS.Signals | null,
    kill(signal?: NodeJS.Signals | number) {
      kills.push(signal)

      return true
    },
    on(event: string, listener: (...args: any[]) => void) {
      emitter.on(event, listener)

      return child
    },
    // A real ChildProcess has both. The unfixed source registers 'error' with
    // `once`, so the fake keeps it too: the P1 tests then fail for the real
    // reason — a spent once-listener — not because a method is missing.
    once(event: string, listener: (...args: any[]) => void) {
      emitter.once(event, listener)

      return child
    },
    removeListener(event: string, listener: (...args: any[]) => void) {
      removed.push(event)
      emitter.removeListener(event, listener)

      return child
    }
  }

  return {
    child,
    kills,
    removed,
    emit(event: string) {
      emitter.emit(event)
    }
  }
}

/** A signal whose abort-listener count a test can read. */
function fakeSignal() {
  const listeners = new Set<() => void>()

  const signal = {
    aborted: false,
    addEventListener(_event: 'abort', listener: () => void) {
      listeners.add(listener)
    },
    removeEventListener(_event: 'abort', listener: () => void) {
      listeners.delete(listener)
    },
    abort() {
      signal.aborted = true

      for (const listener of [...listeners]) {
        listener()
      }
    }
  }

  return { listenerCount: () => listeners.size, signal }
}

describe('killChildOnAbort', () => {
  it('kills at once when the signal is already aborted', () => {
    vi.useFakeTimers()

    try {
      const { child, kills } = fakeChild()

      killChildOnAbort(child, AbortSignal.abort(), 20)

      expect(kills).toEqual(['SIGTERM'])
    } finally {
      vi.useRealTimers()
    }
  })

  it('escalates to SIGKILL when the child has not closed after the grace period', () => {
    vi.useFakeTimers()

    try {
      const { child, kills } = fakeChild()
      const controller = new AbortController()

      killChildOnAbort(child, controller.signal, 20)
      controller.abort()

      expect(kills).toEqual(['SIGTERM'])

      vi.advanceTimersByTime(20)

      expect(kills).toEqual(['SIGTERM', 'SIGKILL'])
    } finally {
      vi.useRealTimers()
    }
  })

  it('does not escalate once the child closes inside the grace period', () => {
    vi.useFakeTimers()

    try {
      const { child, kills, emit } = fakeChild()
      const controller = new AbortController()

      killChildOnAbort(child, controller.signal, 20)
      controller.abort()
      emit('close')

      vi.advanceTimersByTime(1_000)

      expect(kills).toEqual(['SIGTERM'])
    } finally {
      vi.useRealTimers()
    }
  })

  // A child that had already exited when the abort fires carries a non-null
  // exitCode, and its pid may be reused by then. The exit proof is checked
  // BEFORE SIGTERM too — not only before the SIGKILL — so no signal reaches it.
  it('sends no signal at all to a child that already exited (exitCode set)', () => {
    vi.useFakeTimers()

    try {
      const { child, kills } = fakeChild()
      const controller = new AbortController()

      killChildOnAbort(child, controller.signal, 20)

      child.exitCode = 0

      controller.abort()
      vi.advanceTimersByTime(1_000)

      expect(kills).toEqual([])
    } finally {
      vi.useRealTimers()
    }
  })

  it('sends no signal at all to a child that was already signalled (signalCode set)', () => {
    vi.useFakeTimers()

    try {
      const { child, kills } = fakeChild()
      const controller = new AbortController()

      killChildOnAbort(child, controller.signal, 20)

      child.signalCode = 'SIGTERM'

      controller.abort()
      vi.advanceTimersByTime(1_000)

      expect(kills).toEqual([])
    } finally {
      vi.useRealTimers()
    }
  })

  // P1: 'close' waits for the stdio pipes to drain, which a killed git can
  // take time over. Once 'exit' has fired the process is gone and its pid may
  // be reused, so the SIGKILL must not follow. 'exit' is the second proof of
  // exit; teardown drops the escalation on either.
  it('does not escalate once the child exits inside the grace period', () => {
    vi.useFakeTimers()

    try {
      const { child, kills, emit } = fakeChild()
      const controller = new AbortController()

      killChildOnAbort(child, controller.signal, 20)
      controller.abort()
      emit('exit')

      vi.advanceTimersByTime(1_000)

      expect(kills).toEqual(['SIGTERM'])
    } finally {
      vi.useRealTimers()
    }
  })

  // P1: a ChildProcess that has already exited carries a non-null exitCode or
  // signalCode. Signalling it again could hit a reused pid, so the SIGKILL is
  // withheld when either field is set.
  it('withholds the SIGKILL when the child already exited (exitCode set)', () => {
    vi.useFakeTimers()

    try {
      const { child, kills } = fakeChild()
      const controller = new AbortController()

      killChildOnAbort(child, controller.signal, 20)
      controller.abort()

      child.exitCode = 0

      vi.advanceTimersByTime(20)

      expect(kills).toEqual(['SIGTERM'])
    } finally {
      vi.useRealTimers()
    }
  })

  it('withholds the SIGKILL when the child was already signalled (signalCode set)', () => {
    vi.useFakeTimers()

    try {
      const { child, kills } = fakeChild()
      const controller = new AbortController()

      killChildOnAbort(child, controller.signal, 20)
      controller.abort()

      child.signalCode = 'SIGTERM'

      vi.advanceTimersByTime(20)

      expect(kills).toEqual(['SIGTERM'])
    } finally {
      vi.useRealTimers()
    }
  })

  // P1: Node emits 'error' on a kill that fails (EPERM, and other errno values
  // besides ESRCH), not only on a spawn that fails. A once-listener is spent by
  // the first such error, so the SIGKILL's own error would reach no listener —
  // and an unhandled 'error' throws in the Electron main process. The listener
  // must outlive both errors and go only at teardown.
  it('survives an error after SIGTERM and another after SIGKILL', () => {
    vi.useFakeTimers()

    try {
      const { child, kills, emit } = fakeChild()
      const controller = new AbortController()

      child.pid = 4242

      killChildOnAbort(child, controller.signal, 20)
      controller.abort()

      expect(kills).toEqual(['SIGTERM'])

      // The SIGTERM kill failed. The process may still be running, so the
      // escalation stays armed and the error must not throw.
      expect(() => emit('error')).not.toThrow()

      vi.advanceTimersByTime(20)

      expect(kills).toEqual(['SIGTERM', 'SIGKILL'])

      // The SIGKILL kill failed too; still a listener, still no throw.
      expect(() => emit('error')).not.toThrow()
    } finally {
      vi.useRealTimers()
    }
  })

  it('removes its listeners once the child closes', () => {
    const { child, kills, removed, emit } = fakeChild()
    const { listenerCount, signal } = fakeSignal()

    killChildOnAbort(child, signal as unknown as AbortSignal, 20)

    expect(listenerCount()).toBe(1)
    expect(removed).not.toContain('close')

    emit('close')

    expect(listenerCount()).toBe(0)
    expect(removed).toContain('close')
    expect(removed).toContain('exit')

    // A signal that fires after the child is gone must not kill it again.
    signal.abort()

    expect(kills).toEqual([])
  })

  it('removes its listeners once the child errors', () => {
    const { child, kills, removed, emit } = fakeChild()
    const { listenerCount, signal } = fakeSignal()

    killChildOnAbort(child, signal as unknown as AbortSignal, 20)

    emit('error')

    expect(listenerCount()).toBe(0)
    expect(removed).toContain('error')

    signal.abort()

    expect(kills).toEqual([])
  })

  // P2: Node emits 'error' both when a spawn fails and when a kill fails. A
  // failed kill does NOT prove that the process exited, so a spawned child
  // (pid set) that errors after SIGTERM must keep its SIGKILL escalation and
  // keep listening for 'close', which is the only proof of exit.
  it('keeps the SIGKILL escalation when a spawned child emits error after SIGTERM', () => {
    vi.useFakeTimers()

    try {
      const { child, kills, emit } = fakeChild()
      const controller = new AbortController()

      child.pid = 4242

      killChildOnAbort(child, controller.signal, 20)
      controller.abort()

      expect(kills).toEqual(['SIGTERM'])

      emit('error')

      vi.advanceTimersByTime(20)

      expect(kills).toEqual(['SIGTERM', 'SIGKILL'])
    } finally {
      vi.useRealTimers()
    }
  })

  // P2: 'error' proves the process is gone only when the spawn never happened.
  // A child with no pid is torn down at once and no signal ever reaches it.
  it('tears down a child that errors before it spawned (no pid)', () => {
    const { child, kills, removed, emit } = fakeChild()
    const { listenerCount, signal } = fakeSignal()

    killChildOnAbort(child, signal as unknown as AbortSignal, 20)

    emit('error')

    expect(listenerCount()).toBe(0)
    expect(removed).toContain('error')
    expect(removed).toContain('close')

    signal.abort()

    expect(kills).toEqual([])
  })

  // Fix 5: onAbort runs inside controller.abort(), where a throw escapes to
  // the caller and can wedge the probe's timeout callback.
  it('does not let a throw from kill escape the abort listener', () => {
    const child = {
      kill() {
        throw new Error('ESRCH')
      },
      on() {
        return child
      },
      once() {
        return child
      },
      removeListener() {
        return child
      }
    }

    expect(() => killChildOnAbort(child, AbortSignal.abort(), 20)).not.toThrow()

    const controller = new AbortController()

    killChildOnAbort(child, controller.signal, 20)

    expect(() => controller.abort()).not.toThrow()
  })

  // Fix 5: a child that closes the instant it is signalled needs no SIGKILL.
  // The escalation timer is armed only when the child has not already closed.
  it('does not arm the SIGKILL timer when the child closes during the SIGTERM', () => {
    vi.useFakeTimers()

    try {
      const kills: Array<NodeJS.Signals | number | undefined> = []
      const listeners = new Map<string, Array<() => void>>()

      const child = {
        kill(signal?: NodeJS.Signals | number) {
          kills.push(signal)

          for (const listener of [...(listeners.get('close') ?? [])]) {
            listener()
          }

          return true
        },
        on(event: string, listener: () => void) {
          listeners.set(event, [...(listeners.get(event) ?? []), listener])

          return child
        },
        once(event: string, listener: () => void) {
          listeners.set(event, [...(listeners.get(event) ?? []), listener])

          return child
        },
        removeListener(event: string, _listener: () => void) {
          listeners.delete(event)

          return child
        }
      }

      killChildOnAbort(child, AbortSignal.abort(), 20)

      expect(kills).toEqual(['SIGTERM'])

      vi.advanceTimersByTime(1_000)

      expect(kills).toEqual(['SIGTERM'])
    } finally {
      vi.useRealTimers()
    }
  })

  // #125243: a lazy-fetching git spawns fetch/index-pack/pack-objects as its
  // OWN children; killing only the top pid strands them reparented to PID 1.
  // On POSIX the probe spawns git detached (own process group), so the abort
  // signals the whole group by negative pgid alongside the SIGTERM.
  it('signals the process group on abort so git descendants die too (POSIX)', () => {
    vi.useFakeTimers()

    try {
      const { child } = fakeChild()
      const groupKills: Array<[number, string]> = []
      const treeKills: number[] = []

      child.pid = 4242

      killChildOnAbort(child, AbortSignal.abort(), 20, {
        isWindows: false,
        killGroup: (pgid, signal) => groupKills.push([pgid, signal]),
        forceKillProcessTree: pid => treeKills.push(pid)
      })

      expect(groupKills).toEqual([[-4242, 'SIGKILL']])
      // POSIX uses the group signal; the ancestry walk is Windows-only.
      expect(treeKills).toEqual([])

      vi.advanceTimersByTime(20)

      // The SIGKILL escalation re-signals the group, not just the top pid.
      expect(groupKills).toEqual([
        [-4242, 'SIGKILL'],
        [-4242, 'SIGKILL']
      ])
    } finally {
      vi.useRealTimers()
    }
  })

  // Windows has no process groups; the abort tree-kills by pid ancestry
  // (taskkill /T /F) so the managed git host's descendants cannot survive.
  it('tree-kills by pid ancestry on abort (Windows)', () => {
    vi.useFakeTimers()

    try {
      const { child } = fakeChild()
      const treeKills: number[] = []

      child.pid = 4242

      killChildOnAbort(child, AbortSignal.abort(), 20, {
        isWindows: true,
        forceKillProcessTree: pid => treeKills.push(pid)
      })

      expect(treeKills).toEqual([4242])

      vi.advanceTimersByTime(20)

      // The SIGKILL escalation re-walks the tree.
      expect(treeKills).toEqual([4242, 4242])
    } finally {
      vi.useRealTimers()
    }
  })

  // A child without a pid never spawned: no group or tree signal may go out
  // (a negative-pgid send with an undefined pid would signal an unrelated group).
  it('sends no group or tree signal for a child that never spawned (no pid)', () => {
    vi.useFakeTimers()

    try {
      const { child } = fakeChild()
      const groupKills: Array<[number, string]> = []
      const treeKills: number[] = []

      killChildOnAbort(child, AbortSignal.abort(), 20, {
        isWindows: false,
        killGroup: (pgid, signal) => groupKills.push([pgid, signal]),
        forceKillProcessTree: pid => treeKills.push(pid)
      })

      expect(groupKills).toEqual([])
      expect(treeKills).toEqual([])
    } finally {
      vi.useRealTimers()
    }
  })

  // The group send is best-effort at the SIGTERM rung: a group that is
  // already gone (ESRCH) must not stop the SIGKILL escalation from firing.
  it('keeps the SIGKILL escalation when the group signal throws', () => {
    vi.useFakeTimers()

    try {
      const { child, kills } = fakeChild()
      let groupSends = 0

      child.pid = 4242

      killChildOnAbort(
        child,
        AbortSignal.abort(),
        20,
        {
          isWindows: false,
          killGroup: () => {
            groupSends += 1
            throw new Error('ESRCH')
          }
        }
      )

      expect(groupSends).toBe(1)

      vi.advanceTimersByTime(20)

      // The top-pid escalation still fires when the group send throws.
      expect(kills).toEqual(['SIGTERM', 'SIGKILL'])
      expect(groupSends).toBe(2)
    } finally {
      vi.useRealTimers()
    }
  })
})
