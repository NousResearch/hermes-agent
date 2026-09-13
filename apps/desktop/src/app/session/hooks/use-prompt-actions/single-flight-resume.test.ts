import { afterEach, describe, expect, it, vi } from 'vitest'

import {
  clearSingleFlightSessionResumeState,
  registerRecoveredRuntime,
  singleFlightSessionResume,
  takeRecoveredRuntime
} from './single-flight-resume'
import { resumeStoredRuntimeSession, SessionRecoveryAborted, withSessionNotFoundResume } from './utils'

afterEach(() => {
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

  it('does not coalesce the same stored id across backend owners', async () => {
    let release!: () => void
    const pending = new Promise<void>(resolve => (release = resolve))

    const runA = vi.fn(async () => {
      await pending

      return { session_id: 'runtime-a' }
    })

    const runB = vi.fn(async () => {
      await pending

      return { session_id: 'runtime-b' }
    })

    const a = singleFlightSessionResume('stored-shared', runA, {
      scope: { connectionId: 'backend-a', profile: 'default' }
    })

    const b = singleFlightSessionResume('stored-shared', runB, {
      scope: { connectionId: 'backend-b', profile: 'default' }
    })

    await vi.waitFor(() => {
      expect(runA).toHaveBeenCalledTimes(1)
      expect(runB).toHaveBeenCalledTimes(1)
    })
    release()

    await expect(Promise.all([a, b])).resolves.toEqual([
      { session_id: 'runtime-a' },
      { session_id: 'runtime-b' }
    ])
  })

  it('coalesces a scoped foreground resume with an unscoped recovery for the same runtime under a non-default profile', async () => {
    // use-session-actions passes the owner scope; submit/rewind recovery and
    // the route resolver pass none. Both dial the same socket, so they must
    // share ONE flight or a wake-up storm mints two runtimes (#91276).
    const run = vi.fn(async () => {
      await new Promise(resolve => setTimeout(resolve, 10))

      return { session_id: 'rt-work' }
    })

    const [scoped, unscoped] = await Promise.all([
      singleFlightSessionResume('stored-work', run, { scope: 'work' }),
      singleFlightSessionResume('stored-work', run)
    ])

    expect(scoped).toEqual({ session_id: 'rt-work' })
    expect(unscoped).toEqual({ session_id: 'rt-work' })
    expect(run).toHaveBeenCalledTimes(1)
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
