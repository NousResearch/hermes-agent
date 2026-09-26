import { describe, expect, it, vi } from 'vitest'

import { killChildOnAbort } from './git-abort'

/**
 * A child-like object: `kill` records the signals, `once`/`removeListener` let
 * a test fire 'close'/'error' and observe teardown. No main.ts source is read.
 */
function fakeChild() {
  const kills: Array<NodeJS.Signals | number | undefined> = []
  const listeners = new Map<string, Array<() => void>>()
  const removed: string[] = []

  const child = {
    kill(signal?: NodeJS.Signals | number) {
      kills.push(signal)

      return true
    },
    once(event: string, listener: () => void) {
      const list = listeners.get(event) ?? []

      list.push(listener)
      listeners.set(event, list)

      return child
    },
    removeListener(event: string, _listener: () => void) {
      removed.push(event)
      listeners.delete(event)

      return child
    }
  }

  return {
    child,
    kills,
    removed,
    emit(event: string) {
      for (const listener of [...(listeners.get(event) ?? [])]) {
        listener()
      }
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

  it('removes its listeners once the child closes', () => {
    const { child, kills, removed, emit } = fakeChild()
    const { listenerCount, signal } = fakeSignal()

    killChildOnAbort(child, signal as unknown as AbortSignal, 20)

    expect(listenerCount()).toBe(1)
    expect(removed).not.toContain('close')

    emit('close')

    expect(listenerCount()).toBe(0)
    expect(removed).toContain('close')

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

  // Fix 5: onAbort runs inside controller.abort(), where a throw escapes to
  // the caller and can wedge the probe's timeout callback.
  it('does not let a throw from kill escape the abort listener', () => {
    const child = {
      kill() {
        throw new Error('ESRCH')
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
})
