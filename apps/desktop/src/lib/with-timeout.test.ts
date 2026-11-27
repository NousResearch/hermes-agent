import { describe, expect, it, vi } from 'vitest'

import { TimeoutError, withPausableTimeout, withTimeout } from './with-timeout'

describe('withTimeout', () => {
  it('rejects with an onTimeout exception instead of letting it escape the timer callback', async () => {
    vi.useFakeTimers()

    try {
      const callbackFailure = new Error('abort callback failed')

      const result = withTimeout(new Promise<never>(() => undefined), 10, 'work timed out', () => {
        throw callbackFailure
      })

      const rejection = expect(result).rejects.toBe(callbackFailure)

      await vi.advanceTimersByTimeAsync(10)
      await rejection
    } finally {
      vi.useRealTimers()
    }
  })
})

describe('withPausableTimeout', () => {
  it('pauses the countdown during a paused stage and resumes with the remaining budget', async () => {
    vi.useFakeTimers()

    try {
      let paused = false
      const listeners = new Set<() => void>()
      const unsubSpy = vi.fn()

      const setPaused = (next: boolean) => {
        paused = next
        listeners.forEach(fn => fn())
      }

      const promise = withPausableTimeout(
        new Promise<never>(() => undefined),
        1_000,
        'boot timed out',
        onChange => {
          listeners.add(onChange)

          return () => {
            listeners.delete(onChange)
            unsubSpy()
          }
        },
        () => paused
      )

      const caught = promise.catch(err => err)

      // Consume 400ms before entering the paused phase (600ms left).
      await vi.advanceTimersByTimeAsync(400)
      setPaused(true)

      // Spend 5,000ms (5x the total budget) while paused: must not time out.
      await vi.advanceTimersByTimeAsync(5_000)

      let settled = false
      void promise.then(
        () => {
          settled = true
        },
        () => {
          settled = true
        }
      )
      await Promise.resolve()
      expect(settled).toBe(false)

      // Resume: 599ms later still pending, 1ms more rejects with TimeoutError.
      setPaused(false)
      await vi.advanceTimersByTimeAsync(599)
      expect(settled).toBe(false)

      await vi.advanceTimersByTimeAsync(1)
      const err = await caught
      expect(err).toBeInstanceOf(TimeoutError)
      expect((err as Error).message).toBe('boot timed out')
      expect(unsubSpy).toHaveBeenCalledTimes(1)
    } finally {
      vi.useRealTimers()
    }
  })

  it('resolves cleanly while paused and unsubscribes its listener', async () => {
    vi.useFakeTimers()

    try {
      let resolvePending!: (value: string) => void

      const pending = new Promise<string>(resolve => {
        resolvePending = resolve
      })

      const unsubSpy = vi.fn()

      const promise = withPausableTimeout(
        pending,
        1_000,
        'boot timed out',
        () => unsubSpy,
        () => true
      )

      await vi.advanceTimersByTimeAsync(10_000)
      resolvePending('ready')

      await expect(promise).resolves.toBe('ready')
      expect(unsubSpy).toHaveBeenCalledTimes(1)
    } finally {
      vi.useRealTimers()
    }
  })

  it('disposes the pause subscription and cancels the timer when aborted', async () => {
    vi.useFakeTimers()

    try {
      const controller = new AbortController()
      const unsubSpy = vi.fn()
      const onTimeout = vi.fn()

      const promise = withPausableTimeout(
        new Promise<never>(() => undefined),
        1_000,
        'boot timed out',
        () => unsubSpy,
        () => false,
        onTimeout,
        controller.signal
      )

      const caught = promise.catch(err => err)

      controller.abort(new Error('unmounted'))
      const err = await caught
      expect((err as Error).message).toBe('unmounted')
      expect(unsubSpy).toHaveBeenCalledTimes(1)

      await vi.advanceTimersByTimeAsync(10_000)
      expect(onTimeout).not.toHaveBeenCalled()
    } finally {
      vi.useRealTimers()
    }
  })
})
