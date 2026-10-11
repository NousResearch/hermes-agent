import { afterEach, describe, expect, it, vi } from 'vitest'

import {
  clearSingleFlightSessionResumeState,
  registerRecoveredRuntime,
  setSessionResumeProfileCountOverride,
  singleFlightSessionResume,
  takeRecoveredRuntime
} from './single-flight-resume'
import { resumeStoredRuntimeSession, SessionRecoveryAborted, withSessionNotFoundResume } from './utils'

afterEach(() => {
  clearSingleFlightSessionResumeState()
  vi.useRealTimers()
  vi.restoreAllMocks()
})

describe('singleFlightSessionResume', () => {
  it('allows a valid resume to settle inside the ordinary gateway request budget', async () => {
    vi.useFakeTimers()
    setSessionResumeProfileCountOverride(() => 1)

    const flight = singleFlightSessionResume(
      'stored-slow',
      () => new Promise<string>(resolve => setTimeout(() => resolve('runtime-slow'), 25_000))
    )

    await vi.advanceTimersByTimeAsync(25_000)

    await expect(flight).resolves.toBe('runtime-slow')
  })

  it('sizes the settlement ceiling from the multi-profile probe ladder, not a single-profile budget', async () => {
    vi.useFakeTimers()
    // Two configured profiles: one 30s probe window each plus the 30s resume
    // RPC (the budget closed PR 96523 derived) — a resume still inside that
    // ladder must NOT be aborted the way a single-profile 35s ceiling would.
    setSessionResumeProfileCountOverride(() => 2)

    const settleAt = 2 * 30_000 + 29_000

    const flight = singleFlightSessionResume(
      'stored-multi-profile',
      () => new Promise<string>(resolve => setTimeout(() => resolve('runtime-late'), settleAt))
    )

    await vi.advanceTimersByTimeAsync(settleAt)

    await expect(flight).resolves.toBe('runtime-late')
  })

  it('a never-settling resume rejects at the derived deadline instead of wedging the slot forever', async () => {
    vi.useFakeTimers()
    setSessionResumeProfileCountOverride(() => 1)

    const flight = singleFlightSessionResume('stored-wedged', () => new Promise<never>(() => undefined))

    flight.catch(() => undefined)

    // 1 profile => probe window (30s) + RPC window (30s).
    await vi.advanceTimersByTimeAsync(60_000)

    await expect(flight).rejects.toThrow('Timed out resuming session stored-wedged')

    // The slot was released: the next caller starts a FRESH attempt.
    const second = singleFlightSessionResume('stored-wedged', async () => ({ session_id: 'rt-fresh' }))

    await expect(second).resolves.toEqual({ session_id: 'rt-fresh' })
  })

  it('adopts a straggler that lands after the deadline into the recovered-runtime cache', async () => {
    vi.useFakeTimers()
    setSessionResumeProfileCountOverride(() => 1)

    let settleStraggler: ((value: { session_id: string }) => void) | null = null

    const flight = singleFlightSessionResume(
      'stored-straggler',
      () => new Promise<{ session_id: string }>(resolve => (settleStraggler = resolve))
    )

    flight.catch(() => undefined)

    await vi.advanceTimersByTimeAsync(60_000)

    await expect(flight).rejects.toThrow('Timed out resuming session stored-straggler')

    // Nothing cached yet: the straggler has not landed.
    expect(takeRecoveredRuntime('stored-straggler')).toBeUndefined()

    // The gateway finished the resume after the client gave up — the runtime
    // it minted is real, and must be adopted rather than orphaned (#96522).
    settleStraggler!({ session_id: 'rt-straggler' })
    await vi.advanceTimersByTimeAsync(0)

    expect(takeRecoveredRuntime('stored-straggler')).toBe('rt-straggler')
  })

  it('does NOT cache a straggler when a newer flight already owns the stored id', async () => {
    vi.useFakeTimers()
    setSessionResumeProfileCountOverride(() => 1)

    let settleFirst: ((value: { session_id: string }) => void) | null = null
    let settleSecond: ((value: { session_id: string }) => void) | null = null

    const first = singleFlightSessionResume(
      'stored-superseded',
      () => new Promise<{ session_id: string }>(resolve => (settleFirst = resolve))
    )

    first.catch(() => undefined)

    await vi.advanceTimersByTimeAsync(60_000)

    await expect(first).rejects.toThrow('Timed out resuming session stored-superseded')

    // A retry owns the slot and is STILL RUNNING when the first attempt's
    // straggler lands: the retry's caller will adopt its own result, so the
    // old straggler must not overwrite it in the cache.
    const second = singleFlightSessionResume(
      'stored-superseded',
      () => new Promise<{ session_id: string }>(resolve => (settleSecond = resolve))
    )

    second.catch(() => undefined)

    settleFirst!({ session_id: 'rt-first-straggler' })
    await vi.advanceTimersByTimeAsync(0)

    expect(takeRecoveredRuntime('stored-superseded')).toBeUndefined()

    settleSecond!({ session_id: 'rt-second' })
    await expect(second).resolves.toEqual({ session_id: 'rt-second' })
  })

  it('two concurrent resume callers for the same stored id produce ONE session.resume RPC', async () => {
    const requestGateway = vi.fn(async (method: string) => {
      expect(method).toBe('session.resume')
      // Yield so both callers are in flight before either resolves.
      await new Promise(resolve => setTimeout(resolve, 10))

      return { session_id: 'rt-fresh' }
    })

    const deps = { requestGateway: requestGateway as never, resolveProfile: async () => undefined }

    const [a, b] = await Promise.all([
      resumeStoredRuntimeSession('stored-a', deps),
      resumeStoredRuntimeSession('stored-a', deps)
    ])

    expect(a).toBe('rt-fresh')
    expect(b).toBe('rt-fresh')
    expect(requestGateway).toHaveBeenCalledTimes(1)
  })

  it('different stored ids still resume independently', async () => {
    const requestGateway = vi.fn(async (_method: string, params?: Record<string, unknown>) => {
      await new Promise(resolve => setTimeout(resolve, 5))

      return { session_id: `rt-${String(params?.session_id)}` }
    })

    const deps = { requestGateway: requestGateway as never, resolveProfile: async () => undefined }

    const [a, b] = await Promise.all([
      resumeStoredRuntimeSession('stored-a', deps),
      resumeStoredRuntimeSession('stored-b', deps)
    ])

    expect(a).toBe('rt-stored-a')
    expect(b).toBe('rt-stored-b')
    expect(requestGateway).toHaveBeenCalledTimes(2)
  })

  it('a rejected flight is not cached: the next caller retries', async () => {
    const run = vi
      .fn<() => Promise<{ session_id: string }>>()
      .mockRejectedValueOnce(new Error('boom'))
      .mockResolvedValueOnce({ session_id: 'rt-second' })

    await expect(singleFlightSessionResume('stored-a', run)).rejects.toThrow('boom')
    await expect(singleFlightSessionResume('stored-a', run)).resolves.toEqual({ session_id: 'rt-second' })
    expect(run).toHaveBeenCalledTimes(2)
  })
})

describe('drift-abort recovered-runtime cache', () => {
  it('drift-abort does not strand the recovered runtime — it is registered in the cache', async () => {
    const requestGateway = vi.fn(async (method: string) => {
      if (method === 'session.resume') {
        return { session_id: 'rt-recovered' }
      }

      throw new Error('unexpected call')
    })

    const call = vi.fn(async (liveId: string) => {
      if (liveId === 'rt-dead') {
        throw new Error('session not found: rt-dead')
      }

      return 'ok'
    })

    await expect(
      withSessionNotFoundResume('rt-dead', 'stored-a', call, {
        requestGateway: requestGateway as never,
        resolveProfile: async () => undefined,
        driftReason: () => 'user switched away'
      })
    ).rejects.toThrow(SessionRecoveryAborted)

    // The freshly-minted runtime is NOT abandoned: the next action reuses it.
    expect(takeRecoveredRuntime('stored-a')).toBe('rt-recovered')
    // Take-semantics: consumed exactly once.
    expect(takeRecoveredRuntime('stored-a')).toBeUndefined()
  })

  it('a later non-drifted recovery adopts the cached runtime instead of resuming again', async () => {
    registerRecoveredRuntime('stored-a', 'rt-cached')

    const requestGateway = vi.fn(async () => {
      throw new Error('session.resume must not be called when a cached runtime exists')
    })

    const onRecovered = vi.fn()

    const call = vi.fn(async (liveId: string) => {
      if (liveId === 'rt-dead') {
        throw new Error('session not found: rt-dead')
      }

      return `ran-on-${liveId}`
    })

    const outcome = await withSessionNotFoundResume('rt-dead', 'stored-a', call, {
      requestGateway: requestGateway as never,
      resolveProfile: async () => undefined,
      onRecovered
    })

    expect(outcome).toEqual({ recovered: true, result: 'ran-on-rt-cached', sessionId: 'rt-cached' })
    expect(onRecovered).toHaveBeenCalledWith('rt-cached')
    expect(requestGateway).not.toHaveBeenCalled()
  })

  it('takeRecoveredRuntime skips a cached id the caller already knows is dead', () => {
    registerRecoveredRuntime('stored-a', 'rt-dead')

    expect(takeRecoveredRuntime('stored-a', 'rt-dead')).toBeUndefined()
    expect(takeRecoveredRuntime('stored-a')).toBeUndefined()
  })
})
