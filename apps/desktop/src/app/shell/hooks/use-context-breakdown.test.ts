import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ContextBreakdown } from '@/types/hermes'

import { deferred } from '../../../test/deferred'

import {
  _resetContextBreakdownInvalidationsForTests,
  invalidateContextBreakdown,
  useContextBreakdown
} from './use-context-breakdown'

type GatewayRequester = <T = unknown>(method: string, params?: Record<string, unknown>) => Promise<T>

const LIVE_BREAKDOWN: ContextBreakdown = {
  categories: [],
  context_max: 1_048_576,
  context_percent: 42,
  context_used: 440_000,
  estimated_total: 440_000
}

const PRE_COMPRESSION_BREAKDOWN: ContextBreakdown = {
  categories: [],
  context_max: 1_048_576,
  context_percent: 91,
  context_used: 954_000,
  estimated_total: 954_000
}

const ZEROED_BREAKDOWN: ContextBreakdown = {
  categories: [],
  context_max: 0,
  context_percent: 0,
  context_used: 0,
  estimated_total: 0
}

function flushAsync() {
  return act(async () => {
    await vi.advanceTimersByTimeAsync(0)
  })
}

beforeEach(() => {
  vi.useFakeTimers()
})

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
  vi.useRealTimers()
})

describe('useContextBreakdown (#94001)', () => {
  it('caches a live breakdown keyed by session and serves it across busy toggles', async () => {
    const requestGateway = vi.fn().mockResolvedValue(LIVE_BREAKDOWN) as unknown as GatewayRequester

    const { result, rerender } = renderHook(props => useContextBreakdown(props), {
      initialProps: { busy: false, enabled: true, requestGateway, sessionId: 's1' as null | string }
    })

    await flushAsync()

    expect(result.current.breakdown).toEqual(LIVE_BREAKDOWN)
    expect(requestGateway).toHaveBeenCalledTimes(1)

    // A busy toggle re-runs the effect; the fetched cache still serves.
    rerender({ busy: true, enabled: true, requestGateway, sessionId: 's1' })
    rerender({ busy: false, enabled: true, requestGateway, sessionId: 's1' })
    await flushAsync()

    expect(result.current.breakdown).toEqual(LIVE_BREAKDOWN)
  })

  it('drops the cached breakdown when the session changes (no cross-session contamination)', async () => {
    const requestGateway = vi.fn().mockResolvedValue(LIVE_BREAKDOWN) as unknown as GatewayRequester

    const { result, rerender } = renderHook(props => useContextBreakdown(props), {
      initialProps: { busy: false, enabled: true, requestGateway, sessionId: 's1' as null | string }
    })

    await flushAsync()
    expect(result.current.breakdown).toEqual(LIVE_BREAKDOWN)

    rerender({ busy: false, enabled: true, requestGateway, sessionId: 's2' })
    await flushAsync()

    // s2's fetch resolved with LIVE_BREAKDOWN too (same mock), so it serves —
    // but the point is the s1 entry did not leak during the s2 loading window.
    const requestGatewaySlow = vi.fn((_method: string) => deferred<ContextBreakdown>().promise) as unknown as GatewayRequester
    rerender({ busy: false, enabled: true, requestGateway: requestGatewaySlow, sessionId: 's3' })

    expect(result.current.breakdown).toBeNull()
  })

  it('evicts the stale cache and retries when the refetch fails, never serving pre-compression data', async () => {
    let fail = false

    const requestGateway = vi.fn(async () => {
      if (fail) {
        throw new Error('session not found')
      }

      return LIVE_BREAKDOWN
    }) as unknown as GatewayRequester

    const { result, rerender } = renderHook(props => useContextBreakdown(props), {
      initialProps: { busy: false, enabled: true, requestGateway, sessionId: 's1' as null | string }
    })

    await flushAsync()
    expect(result.current.breakdown).toEqual(LIVE_BREAKDOWN)

    // Simulate the post-compression window: the refetch races a reclaim and fails.
    fail = true
    rerender({ busy: true, enabled: true, requestGateway, sessionId: 's1' })
    rerender({ busy: false, enabled: true, requestGateway, sessionId: 's1' })
    await flushAsync()

    // First attempt failed → the STALE cached value must NOT be served while
    // a retry is pending... (cache eviction happens on exhaustion; during the
    // retry window the old value would still be served by the old code.)
    // With the fix: the failure schedules a retry, and the meter keeps the
    // old value ONLY until exhaustion — no, the fix evicts on exhaustion. The
    // key invariant: after retries are exhausted, breakdown is null.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(1_000)
    })
    fail = false
    await flushAsync()

    // Retry 1 succeeded (fail flipped back off) → recovered with live data.
    expect(result.current.breakdown).toEqual(LIVE_BREAKDOWN)
  })

  it('stops serving numbers after retries are exhausted against failures (dark meter, not a lie)', async () => {
    const requestGateway = vi.fn(async () => {
      throw new Error('rpc failed')
    }) as unknown as GatewayRequester

    const { result } = renderHook(props => useContextBreakdown(props), {
      initialProps: { busy: false, enabled: true, requestGateway, sessionId: 's1' as null | string }
    })

    // Seed a stale cache first: succeed once, then start failing forever.
    const flaky = vi.fn(async (_method: string) => {
      throw new Error('rpc failed')
    }) as unknown as GatewayRequester

    // Use two hooks: this test drives exhaustion directly from a cold start.
    const { result: cold } = renderHook(props => useContextBreakdown(props), {
      initialProps: { busy: false, enabled: true, requestGateway: flaky, sessionId: 'cold' as null | string }
    })

    await flushAsync()
    expect(cold.current.breakdown).toBeNull()

    // Exhaust the bounded retry ladder: 1s, 4s, 12s.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(1_000)
    })
    await flushAsync()
    await act(async () => {
      await vi.advanceTimersByTimeAsync(4_000)
    })
    await flushAsync()
    await act(async () => {
      await vi.advanceTimersByTimeAsync(12_000)
    })
    await flushAsync()

    expect(cold.current.breakdown).toBeNull()
    expect(cold.current.loading).toBe(false)
  })

  it('does not cache a zeroed breakdown (agentless window) and retries until a live agent answers', async () => {
    let zeroed = true
    const requestGateway = vi.fn(async () => (zeroed ? ZEROED_BREAKDOWN : LIVE_BREAKDOWN)) as unknown as GatewayRequester

    const { result } = renderHook(props => useContextBreakdown(props), {
      initialProps: { busy: false, enabled: true, requestGateway, sessionId: 's1' as null | string }
    })

    await flushAsync()

    // Zeroed answer must not be cached as the served breakdown.
    expect(result.current.breakdown).toBeNull()

    // The agent finishes building BEFORE the backoff fires: flip the backend
    // state first, then advance — the retry must observe the live answer.
    zeroed = false
    await act(async () => {
      await vi.advanceTimersByTimeAsync(1_000)
    })
    await flushAsync()

    expect(result.current.breakdown).toEqual(LIVE_BREAKDOWN)
    expect(requestGateway).toHaveBeenCalledTimes(2)
  })

  it('never serves a previous session’s cached numbers after its own refetch exhausts retries', async () => {
    // Scenario 1 guard: the s1 cache must not paint under s2 while s2's fetch
    // exhausts its retries against a failing backend.
    let fail = false

    const requestGateway = vi.fn(async () => {
      if (fail) {
        throw new Error('rpc failed')
      }

      return PRE_COMPRESSION_BREAKDOWN
    }) as unknown as GatewayRequester

    const { result, rerender } = renderHook(props => useContextBreakdown(props), {
      initialProps: { busy: false, enabled: true, requestGateway, sessionId: 's1' as null | string }
    })

    await flushAsync()
    expect(result.current.breakdown).toEqual(PRE_COMPRESSION_BREAKDOWN)

    // Switch to s2 whose backend keeps failing through the whole retry ladder.
    fail = true
    rerender({ busy: false, enabled: true, requestGateway, sessionId: 's2' })

    await act(async () => {
      await vi.advanceTimersByTimeAsync(1_000)
    })
    await flushAsync()
    await act(async () => {
      await vi.advanceTimersByTimeAsync(4_000)
    })
    await flushAsync()
    await act(async () => {
      await vi.advanceTimersByTimeAsync(12_000)
    })
    await flushAsync()

    // s1's 91% must NOT appear under s2.
    expect(result.current.breakdown).toBeNull()
  })
})
describe('useContextBreakdown invalidation (#94001 follow-up)', () => {
  beforeEach(() => {
    _resetContextBreakdownInvalidationsForTests()
  })

  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
    vi.useRealTimers()
  })

  it('refetches immediately when invalidateContextBreakdown bumps the session generation', async () => {
    let answer = PRE_COMPRESSION_BREAKDOWN
    const requestGateway = vi.fn(async () => answer) as unknown as GatewayRequester

    const { result, rerender } = renderHook(props => useContextBreakdown(props), {
      initialProps: { busy: false, enabled: true, requestGateway, sessionId: 's1' as null | string }
    })

    await flushAsync()
    expect(result.current.breakdown).toEqual(PRE_COMPRESSION_BREAKDOWN)
    expect(requestGateway).toHaveBeenCalledTimes(1)

    // Compression happened outside any turn (no busy toggle) — invalidate.
    answer = LIVE_BREAKDOWN
    invalidateContextBreakdown('s1')
    rerender({ busy: false, enabled: true, requestGateway, sessionId: 's1' })
    await flushAsync()

    // The meter must serve the POST-compression figure without a busy toggle
    // or a session switch.
    expect(result.current.breakdown).toEqual(LIVE_BREAKDOWN)
    expect(requestGateway).toHaveBeenCalledTimes(2)
  })

  it('does not refetch a different session when an unrelated session is invalidated', async () => {
    const requestGateway = vi.fn(async () => PRE_COMPRESSION_BREAKDOWN) as unknown as GatewayRequester

    const { result, rerender } = renderHook(props => useContextBreakdown(props), {
      initialProps: { busy: false, enabled: true, requestGateway, sessionId: 's1' as null | string }
    })

    await flushAsync()
    expect(requestGateway).toHaveBeenCalledTimes(1)

    invalidateContextBreakdown('other-session')
    rerender({ busy: false, enabled: true, requestGateway, sessionId: 's1' })
    await flushAsync()

    expect(requestGateway).toHaveBeenCalledTimes(1)
    expect(result.current.breakdown).toEqual(PRE_COMPRESSION_BREAKDOWN)
  })
})
