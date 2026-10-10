import { describe, expect, it, vi } from 'vitest'

import {
  attachPowerResumeRemoteRevalidation,
  POWER_RESUME_REVALIDATION_HOLDOFF_MS,
  POWER_RESUME_TRANSPORT_RESET_TIMEOUT_MS,
  RemoteLivenessTracker,
  revalidateSuspectPooledRemoteBackends
} from './remote-liveness'

describe('revalidateSuspectPooledRemoteBackends (#93910)', () => {
  const descriptor = (baseUrl: string) => ({ baseUrl, mode: 'remote' })

  const remoteEntry = (baseUrl: string) => ({
    connectionPromise: Promise.resolve(descriptor(baseUrl)),
    process: null,
    remoteBaseUrl: baseUrl
  })

  it('retires and rebuilds a dead-tunnel descriptor while leaving a healthy one alone', async () => {
    const entries: Array<[string, ReturnType<typeof remoteEntry>]> = [
      ['conn:ssh-dead::default', remoteEntry('http://127.0.0.1:53101')],
      ['conn:ssh-live::default', remoteEntry('http://127.0.0.1:53102')]
    ]

    const probe = vi.fn(async (connection: { baseUrl?: null | string }) => {
      if (connection.baseUrl === 'http://127.0.0.1:53101') {
        throw new Error('connect ECONNREFUSED 127.0.0.1:53101')
      }

      return { ok: true }
    })

    const retire = vi.fn(async (_poolKey: string) => undefined)
    const rebuild = vi.fn(async (_poolKey: string) => descriptor('http://127.0.0.1:53109'))

    const result = await revalidateSuspectPooledRemoteBackends({
      entries,
      log: vi.fn(),
      probe,
      rebuild,
      retire,
      tracker: new RemoteLivenessTracker()
    })

    expect(retire.mock.calls.map(call => call[0])).toEqual(['conn:ssh-dead::default'])
    expect(rebuild.mock.calls.map(call => call[0])).toEqual(['conn:ssh-dead::default'])
    expect(result).toEqual({ rebuilt: ['conn:ssh-dead::default'], retired: ['conn:ssh-dead::default'] })
  })

  it('retires a dead descriptor on the FIRST failed post-resume probe, not after a failure streak', async () => {
    // The background revalidation policy tolerates REMOTE_LIVENESS_FAILURE_LIMIT
    // consecutive failures before dropping a descriptor. After sleep/wake the
    // SSH master is gone for good — a suspect descriptor that fails one bounded
    // probe must be retired immediately instead of surviving two more rounds.
    const retire = vi.fn(async () => undefined)

    const result = await revalidateSuspectPooledRemoteBackends({
      entries: [['conn:ssh-dead::default', remoteEntry('http://127.0.0.1:53101')]],
      log: vi.fn(),
      probe: vi.fn(async () => {
        throw new Error('socket hang up')
      }),
      rebuild: vi.fn(async () => descriptor('http://127.0.0.1:53110')),
      retire,
      tracker: new RemoteLivenessTracker()
    })

    expect(retire).toHaveBeenCalledTimes(1)
    expect(result.retired).toEqual(['conn:ssh-dead::default'])
  })

  it('skips local child-backed entries entirely', async () => {
    const probe = vi.fn(async () => ({ ok: true }))
    const retire = vi.fn()
    const rebuild = vi.fn()

    const result = await revalidateSuspectPooledRemoteBackends({
      entries: [
        [
          'default',
          {
            connectionPromise: Promise.resolve(descriptor('http://127.0.0.1:9')),
            process: { pid: 4 },
            remoteBaseUrl: null
          }
        ],
        [
          'work',
          {
            connectionPromise: Promise.resolve(descriptor('http://127.0.0.1:9')),
            process: { pid: 5 },
            remoteBaseUrl: ''
          }
        ]
      ],
      log: vi.fn(),
      probe,
      rebuild,
      retire,
      tracker: new RemoteLivenessTracker()
    })

    expect(probe).not.toHaveBeenCalled()
    expect(retire).not.toHaveBeenCalled()
    expect(rebuild).not.toHaveBeenCalled()
    expect(result).toEqual({ rebuilt: [], retired: [] })
  })

  it('fails closed when the rebuild dial rejects: descriptor is retired, no throw, no rebuilt claim', async () => {
    const log = vi.fn()

    const result = await revalidateSuspectPooledRemoteBackends({
      entries: [['conn:ssh-dead::default', remoteEntry('http://127.0.0.1:53101')]],
      log,
      probe: vi.fn(async () => {
        throw new Error('socket hang up')
      }),
      rebuild: vi.fn(async () => {
        throw new Error('ssh bootstrap failed')
      }),
      retire: vi.fn(async () => undefined),
      tracker: new RemoteLivenessTracker()
    })

    expect(result.retired).toEqual(['conn:ssh-dead::default'])
    expect(result.rebuilt).toEqual([])
    expect(log.mock.calls.some(call => String(call[0]).includes('ssh bootstrap failed'))).toBe(true)
  })

  it('does not rebuild on top of a descriptor whose retire failed', async () => {
    const rebuild = vi.fn(async () => descriptor('http://127.0.0.1:53110'))

    const result = await revalidateSuspectPooledRemoteBackends({
      entries: [['conn:ssh-dead::default', remoteEntry('http://127.0.0.1:53101')]],
      log: vi.fn(),
      probe: vi.fn(async () => {
        throw new Error('socket hang up')
      }),
      rebuild,
      retire: vi.fn(async () => {
        throw new Error('stop timed out')
      }),
      tracker: new RemoteLivenessTracker()
    })

    expect(rebuild).not.toHaveBeenCalled()
    expect(result).toEqual({ rebuilt: [], retired: [] })
  })

  it('clears the shared failure streak for a retired base URL so the rebuilt tunnel starts clean', async () => {
    const tracker = new RemoteLivenessTracker()
    tracker.recordFailure('http://127.0.0.1:53101')
    tracker.recordFailure('http://127.0.0.1:53101')

    await revalidateSuspectPooledRemoteBackends({
      entries: [['conn:ssh-dead::default', remoteEntry('http://127.0.0.1:53101')]],
      log: vi.fn(),
      probe: vi.fn(async () => {
        throw new Error('socket hang up')
      }),
      rebuild: vi.fn(async () => descriptor('http://127.0.0.1:53110')),
      retire: vi.fn(async () => undefined),
      tracker
    })

    expect(tracker.recordFailure('http://127.0.0.1:53101')).toEqual({ failures: 1, shouldReset: false })
  })
})

describe('attachPowerResumeRemoteRevalidation (#93910)', () => {
  function fakePowerMonitor() {
    const listeners = new Map<string, Array<() => void>>()

    return {
      emit(event: string) {
        for (const listener of listeners.get(event) ?? []) {
          listener()
        }
      },
      on(event: string, listener: () => void) {
        listeners.set(event, [...(listeners.get(event) ?? []), listener])

        return this
      }
    }
  }

  it('resets before every renderer redial, coalesces wake bursts, and never resets for unlock alone', async () => {
    const order: string[] = []
    let finish!: () => void

    const pending = new Promise<void>(resolve => {
      finish = resolve
    })

    const resetTransports = vi
      .fn()
      .mockImplementationOnce(() => pending)
      .mockResolvedValue(undefined)

    let now = 1_000_000

    const trigger = attachPowerResumeRemoteRevalidation({
      log: vi.fn(),
      now: () => now,
      notifyResume: () => order.push('notify'),
      powerMonitor: fakePowerMonitor(),
      resetTransports,
      revalidate: async () => {
        order.push('sweep')
      }
    })

    const wake = trigger('resume')
    const unlock = trigger('unlock-screen')
    const bounce = trigger('resume')
    await Promise.resolve()
    expect(resetTransports).toHaveBeenCalledTimes(1)
    expect(order).toEqual([])
    finish()
    await Promise.all([wake, unlock, bounce])
    expect(order.filter(step => step === 'notify')).toHaveLength(3)
    expect(order.filter(step => step === 'sweep')).toHaveLength(1)
    expect(order[0]).toBe('notify')

    await trigger('resume')
    expect(order.filter(step => step === 'notify')).toHaveLength(4)
    expect(resetTransports).toHaveBeenCalledTimes(1)
    now += POWER_RESUME_REVALIDATION_HOLDOFF_MS + 1
    await trigger('unlock-screen')
    expect(resetTransports).toHaveBeenCalledTimes(1)
    // Unlock can start a sweep but cannot suppress cleanup for the next wake.
    await trigger('resume')
    expect(resetTransports).toHaveBeenCalledTimes(2)
    expect(order.filter(step => step === 'notify')).toHaveLength(6)
  })

  it('notifies and revalidates after stuck or throwing cleanup, and recovers on a later wake', async () => {
    vi.useFakeTimers()

    try {
      const log = vi.fn()

      const notifyResume = vi.fn().mockImplementationOnce(() => {
        throw new Error('renderer unavailable')
      })

      const revalidate = vi.fn(async () => undefined)
      let rejectLate!: (error: Error) => void

      const resetTransports = vi
        .fn()
        .mockImplementationOnce(
          () =>
            new Promise<void>((_resolve, reject) => {
              rejectLate = reject
            })
        )
        .mockImplementationOnce(() => {
          throw new Error('partition unavailable')
        })
        .mockResolvedValue(undefined)

      const trigger = attachPowerResumeRemoteRevalidation({
        log,
        notifyResume,
        powerMonitor: fakePowerMonitor(),
        resetTransports,
        revalidate
      })

      const stuck = trigger('resume')
      await vi.advanceTimersByTimeAsync(POWER_RESUME_TRANSPORT_RESET_TIMEOUT_MS)
      await stuck
      expect(notifyResume).toHaveBeenCalledTimes(1)
      expect(revalidate).toHaveBeenCalledTimes(1)
      rejectLate(new Error('late failure'))
      await Promise.resolve()

      await vi.advanceTimersByTimeAsync(POWER_RESUME_REVALIDATION_HOLDOFF_MS + 1)
      await trigger('resume')
      expect(notifyResume).toHaveBeenCalledTimes(2)
      expect(revalidate).toHaveBeenCalledTimes(2)
      expect(log.mock.calls.map(call => call[0]).join('\n')).toMatch(
        /timed out[\s\S]*renderer unavailable[\s\S]*partition unavailable/
      )

      await vi.advanceTimersByTimeAsync(POWER_RESUME_REVALIDATION_HOLDOFF_MS + 1)
      await trigger('resume')
      expect(resetTransports).toHaveBeenCalledTimes(3)
      expect(notifyResume).toHaveBeenCalledTimes(3)
      expect(revalidate).toHaveBeenCalledTimes(3)
      expect(vi.getTimerCount()).toBe(0)
    } finally {
      vi.useRealTimers()
    }
  })

  it('kicks one bounded revalidation per resume, coalescing resume + unlock-screen bursts (no hot loop)', async () => {
    const powerMonitor = fakePowerMonitor()
    let resolveRevalidate: (() => void) | undefined

    const revalidate = vi.fn(
      () =>
        new Promise<void>(resolve => {
          resolveRevalidate = resolve
        })
    )

    let now = 1_000_000
    attachPowerResumeRemoteRevalidation({
      log: vi.fn(),
      now: () => now,
      powerMonitor,
      revalidate
    })

    // macOS wake fires 'resume' and 'unlock-screen' near-simultaneously.
    powerMonitor.emit('resume')
    powerMonitor.emit('unlock-screen')
    powerMonitor.emit('resume')
    expect(revalidate).toHaveBeenCalledTimes(1)

    resolveRevalidate?.()
    await Promise.resolve()
    await Promise.resolve()

    // Still inside the holdoff window: no re-kick even after the run settled.
    now += POWER_RESUME_REVALIDATION_HOLDOFF_MS - 1
    powerMonitor.emit('resume')
    expect(revalidate).toHaveBeenCalledTimes(1)

    // A later, distinct wake is allowed through.
    now += POWER_RESUME_REVALIDATION_HOLDOFF_MS
    powerMonitor.emit('resume')
    expect(revalidate).toHaveBeenCalledTimes(2)
  })

  it('swallows and logs a rejected revalidation without wedging future wakes', async () => {
    const powerMonitor = fakePowerMonitor()
    const log = vi.fn()

    const revalidate = vi.fn(async () => {
      throw new Error('probe exploded')
    })

    let now = 5_000_000

    const trigger = attachPowerResumeRemoteRevalidation({
      log,
      now: () => now,
      powerMonitor,
      revalidate
    })

    powerMonitor.emit('resume')
    await trigger()
    expect(log.mock.calls.some(call => String(call[0]).includes('probe exploded'))).toBe(true)

    now += POWER_RESUME_REVALIDATION_HOLDOFF_MS + 1
    powerMonitor.emit('resume')
    expect(revalidate).toHaveBeenCalledTimes(2)
  })
})
