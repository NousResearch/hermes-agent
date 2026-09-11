import { afterEach, describe, expect, it, vi } from 'vitest'

import { isTimeoutError } from '@/lib/with-timeout'

import {
  clearSingleFlightSessionResumeState,
  registerRecoveredRuntime,
  SESSION_RESUME_SETTLEMENT_TIMEOUT_MS,
  singleFlightSessionResume,
  takeRecoveredRuntime
} from './single-flight-resume'
import { resumeStoredRuntimeSession, SessionRecoveryAborted, withSessionNotFoundResume } from './utils'

afterEach(() => {
  vi.useRealTimers()
  clearSingleFlightSessionResumeState()
  vi.restoreAllMocks()
})

describe('singleFlightSessionResume', () => {
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

describe('bounded settlement and straggler adoption', () => {
  it('a never-settling resume rejects at the deadline instead of wedging the slot forever', async () => {
    vi.useFakeTimers()

    const flight = singleFlightSessionResume('stored-wedged', () => new Promise<never>(() => {}))
    const settled = flight.catch(error => error)

    await vi.advanceTimersByTimeAsync(SESSION_RESUME_SETTLEMENT_TIMEOUT_MS)

    expect(isTimeoutError(await settled)).toBe(true)
  })

  it('a later caller for the same stored id gets a FRESH attempt, not the dead flight', async () => {
    vi.useFakeTimers()

    const first = singleFlightSessionResume('stored-wedged', () => new Promise<never>(() => {}))
    const firstSettled = first.catch(() => 'timed-out')

    await vi.advanceTimersByTimeAsync(SESSION_RESUME_SETTLEMENT_TIMEOUT_MS)
    expect(await firstSettled).toBe('timed-out')

    const run = vi.fn(async () => ({ session_id: 'rt-second' }))

    await expect(singleFlightSessionResume('stored-wedged', run)).resolves.toEqual({ session_id: 'rt-second' })
    expect(run).toHaveBeenCalledTimes(1)
  })

  it('a joiner of an already-wedged flight inherits the same deadline', async () => {
    vi.useFakeTimers()

    const first = singleFlightSessionResume('stored-wedged', () => new Promise<never>(() => {}))
    // The joiner passes no run() of its own — it must not hang past the ceiling.
    const joiner = singleFlightSessionResume('stored-wedged', async () => ({ session_id: 'never-used' }))

    expect(joiner).toBe(first)

    const settled = joiner.catch(error => error)

    await vi.advanceTimersByTimeAsync(SESSION_RESUME_SETTLEMENT_TIMEOUT_MS)

    expect(isTimeoutError(await settled)).toBe(true)
  })

  it('a slow-but-legitimate resume (profile probe + RPC) still settles inside the ceiling', async () => {
    vi.useFakeTimers()

    // The worst healthy shape the ceiling is derived from: an active-profile
    // probe, one cross-profile probe, then the resume RPC — each on its own
    // 30s budget. A tighter ceiling would abort this and re-mint a runtime.
    const flight = singleFlightSessionResume(
      'stored-slow',
      () => new Promise<{ session_id: string }>(resolve => setTimeout(() => resolve({ session_id: 'rt-slow' }), 89_000))
    )

    await vi.advanceTimersByTimeAsync(89_000)

    await expect(flight).resolves.toEqual({ session_id: 'rt-slow' })
  })

  it('a runtime minted by a timed-out resume is adopted, not stranded on the gateway', async () => {
    vi.useFakeTimers()

    let land: (value: { session_id: string }) => void = () => {}
    const flight = singleFlightSessionResume(
      'stored-late',
      () =>
        new Promise<{ session_id: string }>(resolve => {
          land = resolve
        })
    )
    const settled = flight.catch(() => 'timed-out')

    await vi.advanceTimersByTimeAsync(SESSION_RESUME_SETTLEMENT_TIMEOUT_MS)
    expect(await settled).toBe('timed-out')

    // The RPC was never cancelled: it lands a real, registered runtime.
    land({ session_id: 'rt-late' })
    await vi.advanceTimersByTimeAsync(0)

    // The next resume-shaped action reuses it instead of minting a second one.
    expect(takeRecoveredRuntime('stored-late')).toBe('rt-late')
  })

  it('a straggler is NOT cached when a newer flight already owns the stored id', async () => {
    vi.useFakeTimers()

    let land: (value: { session_id: string }) => void = () => {}
    const first = singleFlightSessionResume(
      'stored-late',
      () =>
        new Promise<{ session_id: string }>(resolve => {
          land = resolve
        })
    )
    const settled = first.catch(() => 'timed-out')

    await vi.advanceTimersByTimeAsync(SESSION_RESUME_SETTLEMENT_TIMEOUT_MS)
    expect(await settled).toBe('timed-out')

    // A new flight claims the slot before the straggler lands.
    const second = singleFlightSessionResume('stored-late', () => new Promise<never>(() => {}))

    void second.catch(() => undefined)
    land({ session_id: 'rt-late' })
    await vi.advanceTimersByTimeAsync(0)

    // Its caller adopts the second flight's result; caching the older runtime
    // here would aim the next action at the wrong one.
    expect(takeRecoveredRuntime('stored-late')).toBeUndefined()
  })

  it('a straggler that fails minted nothing and caches nothing', async () => {
    vi.useFakeTimers()

    let fail: (error: unknown) => void = () => {}
    const flight = singleFlightSessionResume(
      'stored-late',
      () =>
        new Promise<{ session_id: string }>((_resolve, reject) => {
          fail = reject
        })
    )
    const settled = flight.catch(() => 'timed-out')

    await vi.advanceTimersByTimeAsync(SESSION_RESUME_SETTLEMENT_TIMEOUT_MS)
    expect(await settled).toBe('timed-out')

    fail(new Error('resume failed'))
    await vi.advanceTimersByTimeAsync(0)

    expect(takeRecoveredRuntime('stored-late')).toBeUndefined()
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
