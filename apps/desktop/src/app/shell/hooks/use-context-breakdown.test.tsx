import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ContextBreakdown } from '@/types/hermes'

import { deferred } from '../../../test/deferred'

import { useContextBreakdown } from './use-context-breakdown'

type GatewayRequester = <T = unknown>(method: string, params?: Record<string, unknown>) => Promise<T>

function breakdownOf(contextMax: number): ContextBreakdown {
  return {
    categories: [{ color: 'teal', id: 'conversation', label: 'Conversation', tokens: 24_100 }],
    context_max: contextMax,
    context_percent: 9,
    context_used: 24_100,
    estimated_total: 25_400,
    model: 'test-model'
  }
}

async function flushAsync() {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(0)
  })
}

beforeEach(() => {
  vi.useFakeTimers()
})

afterEach(() => {
  cleanup()
  vi.useRealTimers()
  vi.restoreAllMocks()
})

describe('useContextBreakdown retry', () => {
  it('recovers from a session-not-found rejection once the gateway registers the session', async () => {
    // The gateway 4001s until its runtime registry knows the session — the
    // restart race. One failed fetch, then a real answer.
    const requestGateway = vi
      .fn()
      .mockRejectedValueOnce(new Error('session not found'))
      .mockResolvedValue(breakdownOf(123_000))

    const { result } = renderHook(() =>
      useContextBreakdown({ busy: false, enabled: true, requestGateway, sessionId: 'runtime-1' })
    )

    await flushAsync()

    await act(async () => {
      await vi.advanceTimersByTimeAsync(5_000)
    })

    expect(requestGateway).toHaveBeenCalledTimes(2)
    expect(result.current.breakdown?.context_max).toBe(123_000)
    expect(result.current.loading).toBe(false)
  })

  it('refetches when the first answer has no hydration data instead of painting an empty gauge', async () => {
    // The session is registered but the agent is not hydrated yet: the RPC
    // SUCCEEDS with context_max 0. Accepting that answer would strand the
    // gauge at zero until something else changes.
    const requestGateway = vi.fn().mockResolvedValueOnce(breakdownOf(0)).mockResolvedValue(breakdownOf(999_000))

    const { result } = renderHook(() =>
      useContextBreakdown({ busy: false, enabled: true, requestGateway, sessionId: 'runtime-1' })
    )

    await flushAsync()
    expect(result.current.breakdown).toBeNull()

    await act(async () => {
      await vi.advanceTimersByTimeAsync(5_000)
    })

    expect(requestGateway).toHaveBeenCalledTimes(2)
    expect(result.current.breakdown?.context_max).toBe(999_000)
    expect(result.current.loading).toBe(false)
  })

  it('stops after a bounded number of attempts instead of retrying forever', async () => {
    const requestGateway = vi.fn(async (_method: string) => {
      throw new Error('session not found')
    })

    const { result } = renderHook(() =>
      useContextBreakdown({ busy: false, enabled: true, requestGateway, sessionId: 'runtime-1' })
    )

    await flushAsync()
    expect(requestGateway).toHaveBeenCalledTimes(1)

    await act(async () => {
      await vi.advanceTimersByTimeAsync(60_000)
    })

    // RETRY_LIMIT is 4 retries: 1 initial fetch + 4 retries — then the hook
    // gives up and never schedules another attempt, however long it waits.
    expect(requestGateway).toHaveBeenCalledTimes(5)
    expect(result.current.breakdown).toBeNull()
    expect(result.current.loading).toBe(false)
  })

  it('does not report session A numbers once the focused session is B', async () => {
    const sessionA = deferred<ContextBreakdown>()

    const requestGateway = vi.fn((method: string, params?: Record<string, unknown>) => {
      if (params?.session_id === 'runtime-2') {
        return new Promise<ContextBreakdown>(() => undefined)
      }

      return sessionA.promise
    }) as unknown as GatewayRequester

    const { rerender, result } = renderHook(
      ({ sessionId }) => useContextBreakdown({ busy: false, enabled: true, requestGateway, sessionId }),
      { initialProps: { sessionId: 'runtime-1' } }
    )

    rerender({ sessionId: 'runtime-2' })
    await flushAsync()

    // A's fetch finally lands after the switch; it must not paint under B.
    await act(async () => {
      sessionA.resolve(breakdownOf(123_000))
      await vi.advanceTimersByTimeAsync(0)
    })

    expect(result.current.breakdown).toBeNull()
  })

  it('starts a fresh retry budget after a dependency change mid-retry', async () => {
    let rejectNext = true

    const requestGateway = vi.fn(async (method: string): Promise<unknown> => {
      if (rejectNext) {
        throw new Error('session not found')
      }

      return breakdownOf(123_000)
    }) as unknown as GatewayRequester

    const { rerender, result } = renderHook(
      ({ busy }) => useContextBreakdown({ busy, enabled: true, requestGateway, sessionId: 'runtime-1' }),
      { initialProps: { busy: false } }
    )

    // First run: the attempt burns itself on a failure...
    await flushAsync()
    expect(requestGateway).toHaveBeenCalledTimes(1)

    // ...mid-retry (before the scheduled retry fires) a turn boundary flips
    // `busy` and remounts the effect.
    rerender({ busy: true })
    rerender({ busy: false })

    await act(async () => {
      await vi.advanceTimersByTimeAsync(5_000)
    })

    // The old run's retry timer was cancelled on cleanup; the new run began
    // from attempt 0, so the very next fetch is one it accepts.
    rejectNext = false
    await act(async () => {
      await vi.advanceTimersByTimeAsync(5_000)
    })

    expect(result.current.breakdown?.context_max).toBe(123_000)
  })
})
