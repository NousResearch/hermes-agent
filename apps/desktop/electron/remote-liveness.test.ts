import { describe, expect, it, vi } from 'vitest'

import {
  ensureHealthyPooledRemoteBackendForDispatch,
  HOST_EVENT_DISTINCT_PROFILE_THRESHOLD,
  HOST_EVENT_WINDOW_MS,
  POOLED_REMOTE_DIAL_CONCURRENCY,
  POOLED_REMOTE_DIAL_JITTER_MS,
  POOLED_REMOTE_DISPATCH_PROBE_TIMEOUT_MS,
  PooledRemoteDialGate,
  REMOTE_LIVENESS_FAILURE_LIMIT,
  REMOTE_LIVENESS_FAILURE_WINDOW_MS,
  REMOTE_LIVENESS_TIMEOUT_MS,
  RemoteHostEventTracker,
  RemoteLivenessTracker,
  RemoteRevalidationCoordinator,
  revalidatePooledRemoteBackends,
  revalidateRemoteConnection
} from './remote-liveness'

describe('RemoteLivenessTracker', () => {
  it('requires consecutive failures before resetting a connection', () => {
    const tracker = new RemoteLivenessTracker()

    for (let failures = 1; failures < REMOTE_LIVENESS_FAILURE_LIMIT; failures += 1) {
      expect(tracker.recordFailure('https://gateway.example.com')).toEqual({ failures, shouldReset: false })
    }

    expect(tracker.recordFailure('https://gateway.example.com')).toEqual({
      failures: REMOTE_LIVENESS_FAILURE_LIMIT,
      shouldReset: true
    })
  })

  it('clears a failure streak after a successful probe', () => {
    const tracker = new RemoteLivenessTracker()

    tracker.recordFailure('https://gateway.example.com')
    tracker.recordFailure('https://gateway.example.com')
    tracker.recordSuccess('https://gateway.example.com')

    expect(tracker.recordFailure('https://gateway.example.com')).toEqual({ failures: 1, shouldReset: false })
  })

  it('tracks different gateways independently', () => {
    const tracker = new RemoteLivenessTracker(2)

    expect(tracker.recordFailure('https://one.example.com')).toEqual({ failures: 1, shouldReset: false })
    expect(tracker.recordFailure('https://two.example.com')).toEqual({ failures: 1, shouldReset: false })
    expect(tracker.recordFailure('https://one.example.com')).toEqual({ failures: 2, shouldReset: true })
    expect(tracker.recordFailure('https://two.example.com')).toEqual({ failures: 2, shouldReset: true })
  })

  it('clears only the successful gateway streak', () => {
    const tracker = new RemoteLivenessTracker(3)

    tracker.recordFailure('https://one.example.com')
    tracker.recordFailure('https://two.example.com')
    tracker.recordSuccess('https://one.example.com')

    expect(tracker.recordFailure('https://one.example.com')).toEqual({ failures: 1, shouldReset: false })
    expect(tracker.recordFailure('https://two.example.com')).toEqual({ failures: 2, shouldReset: false })
  })

  it('does not accumulate isolated failures across separate reconnect episodes', () => {
    let now = 0
    const tracker = new RemoteLivenessTracker(3, REMOTE_LIVENESS_FAILURE_WINDOW_MS, () => now)

    expect(tracker.recordFailure('https://gateway.example.com')).toEqual({ failures: 1, shouldReset: false })
    now += REMOTE_LIVENESS_FAILURE_WINDOW_MS + 1
    expect(tracker.recordFailure('https://gateway.example.com')).toEqual({ failures: 1, shouldReset: false })
  })

  it('clears all failure streaks when the connection state resets', () => {
    const tracker = new RemoteLivenessTracker(3)

    tracker.recordFailure('https://one.example.com')
    tracker.recordFailure('https://two.example.com')
    tracker.clear()

    expect(tracker.recordFailure('https://one.example.com')).toEqual({ failures: 1, shouldReset: false })
    expect(tracker.recordFailure('https://two.example.com')).toEqual({ failures: 1, shouldReset: false })
  })

  it('starts a fresh streak after the reset threshold is consumed', () => {
    const tracker = new RemoteLivenessTracker(1)

    expect(tracker.recordFailure('https://gateway.example.com')).toEqual({ failures: 1, shouldReset: true })
    expect(tracker.recordFailure('https://gateway.example.com')).toEqual({ failures: 1, shouldReset: true })
  })

  it('rejects invalid failure limits', () => {
    expect(() => new RemoteLivenessTracker(0)).toThrow(/positive integer/i)
    expect(() => new RemoteLivenessTracker(1.5)).toThrow(/positive integer/i)
    expect(() => new RemoteLivenessTracker(1, 0)).toThrow(/window must be positive/i)
  })
})

describe('RemoteRevalidationCoordinator', () => {
  it('coalesces simultaneous probes for the same cached connection', async () => {
    const coordinator = new RemoteRevalidationCoordinator()
    const connection = Promise.resolve({ baseUrl: 'https://gateway.example.com' })
    let resolveProbe: (value: string) => void = () => undefined

    const probe = vi.fn(
      () =>
        new Promise<string>(resolve => {
          resolveProbe = resolve
        })
    )

    const first = coordinator.run(connection, probe)
    const second = coordinator.run(connection, probe)
    const third = coordinator.run(connection, probe)

    await Promise.resolve()

    expect(second).toBe(first)
    expect(third).toBe(first)
    expect(probe).toHaveBeenCalledOnce()

    resolveProbe('healthy')
    await expect(Promise.all([first, second, third])).resolves.toEqual(['healthy', 'healthy', 'healthy'])
  })

  it('runs a fresh probe after the prior one settles', async () => {
    const coordinator = new RemoteRevalidationCoordinator()
    const connection = Promise.resolve({ baseUrl: 'https://gateway.example.com' })
    const probe = vi.fn().mockResolvedValue('healthy')

    await coordinator.run(connection, probe)
    await coordinator.run(connection, probe)

    expect(probe).toHaveBeenCalledTimes(2)
  })

  it('does not coalesce different cached connections', async () => {
    const coordinator = new RemoteRevalidationCoordinator()
    const probe = vi.fn().mockResolvedValue('healthy')

    await Promise.all([coordinator.run(Promise.resolve('one'), probe), coordinator.run(Promise.resolve('two'), probe)])

    expect(probe).toHaveBeenCalledTimes(2)
  })

  it('cleans up a rejected probe so it can be retried', async () => {
    const coordinator = new RemoteRevalidationCoordinator()
    const connection = Promise.resolve({ baseUrl: 'https://gateway.example.com' })
    const probe = vi.fn().mockRejectedValueOnce(new Error('offline')).mockResolvedValueOnce('healthy')

    await expect(coordinator.run(connection, probe)).rejects.toThrow('offline')
    await expect(coordinator.run(connection, probe)).resolves.toBe('healthy')
    expect(probe).toHaveBeenCalledTimes(2)
  })
})

describe('revalidateRemoteConnection', () => {
  function harness(overrides: Record<string, unknown> = {}) {
    const connection = { baseUrl: 'https://gateway.example.com/', mode: 'remote' }
    const connectionPromise = Promise.resolve(connection)
    const current = { promise: connectionPromise as null | Promise<typeof connection> }
    const log = vi.fn()
    const probe = vi.fn().mockResolvedValue({ ok: true })
    const resetConnection = vi.fn()
    const tracker = new RemoteLivenessTracker()

    return {
      connectionPromise,
      current,
      log,
      options: {
        connectionPromise,
        currentConnectionPromise: () => current.promise,
        log,
        probe,
        resetConnection,
        tracker,
        ...overrides
      },
      probe,
      resetConnection,
      tracker
    }
  }

  it('probes the normalized status URL with the production timeout', async () => {
    const test = harness()

    await expect(revalidateRemoteConnection(test.options)).resolves.toEqual({ ok: true, rebuilt: false })
    expect(test.probe).toHaveBeenCalledWith(
      expect.objectContaining({ baseUrl: 'https://gateway.example.com/' }),
      '/api/status',
      {
        timeoutMs: REMOTE_LIVENESS_TIMEOUT_MS
      }
    )
    expect(test.resetConnection).not.toHaveBeenCalled()
  })

  it('keeps failures one and two, then resets on the third failure', async () => {
    const probe = vi.fn().mockRejectedValue(new Error('offline'))
    const test = harness({ probe })

    await expect(revalidateRemoteConnection(test.options)).resolves.toEqual({ ok: true, rebuilt: false })
    await expect(revalidateRemoteConnection(test.options)).resolves.toEqual({ ok: true, rebuilt: false })
    await expect(revalidateRemoteConnection(test.options)).resolves.toEqual({ ok: true, rebuilt: true })

    expect(probe).toHaveBeenCalledTimes(3)
    expect(test.resetConnection).toHaveBeenCalledOnce()
    expect(test.log).toHaveBeenNthCalledWith(1, expect.stringContaining('(1/3)'))
    expect(test.log).toHaveBeenNthCalledWith(2, expect.stringContaining('(2/3)'))
    expect(test.log).toHaveBeenLastCalledWith(expect.stringContaining('dropping stale connection'))
  })

  it('ignores a late failed probe after the cached connection is replaced', async () => {
    let rejectProbe: (error: Error) => void = () => undefined

    const probe = vi.fn(
      () =>
        new Promise((_resolve, reject) => {
          rejectProbe = reject
        })
    )

    const test = harness({ probe })
    const pending = revalidateRemoteConnection(test.options)

    await Promise.resolve()
    test.current.promise = Promise.resolve({ baseUrl: 'https://new.example.com', mode: 'remote' })
    rejectProbe(new Error('old connection failed'))

    await expect(pending).resolves.toEqual({ ok: true, rebuilt: false })
    expect(test.resetConnection).not.toHaveBeenCalled()
    expect(test.log).not.toHaveBeenCalled()
    expect(test.tracker.recordFailure('https://gateway.example.com')).toEqual({ failures: 1, shouldReset: false })
  })

  it('does not probe a local, rejected, or already replaced connection', async () => {
    const replaced = harness()

    replaced.current.promise = null
    await expect(revalidateRemoteConnection(replaced.options)).resolves.toEqual({ ok: true, rebuilt: false })
    expect(replaced.probe).not.toHaveBeenCalled()

    const localConnection = { baseUrl: 'http://127.0.0.1:3000', mode: 'local' }
    const localPromise = Promise.resolve(localConnection)

    const local = harness({
      connectionPromise: localPromise,
      currentConnectionPromise: () => localPromise
    })

    await expect(revalidateRemoteConnection(local.options)).resolves.toEqual({ ok: true, rebuilt: false })
    expect(local.probe).not.toHaveBeenCalled()

    const rejectedPromise = Promise.reject(new Error('boot failed'))

    const rejected = harness({
      connectionPromise: rejectedPromise,
      currentConnectionPromise: () => rejectedPromise
    })

    await expect(revalidateRemoteConnection(rejected.options)).resolves.toEqual({ ok: true, rebuilt: false })
    expect(rejected.probe).not.toHaveBeenCalled()
  })

  // The primary connection has no pool sibling to vote with, so it cannot use
  // the pooled host-event signal. A busy host still makes its quick streak
  // expire, and dropping it is a visible whole-app reload.
  describe('drop confirmation', () => {
    function confirmation(overrides: Record<string, unknown> = {}) {
      return {
        backoff: vi.fn(async () => undefined),
        transportAlive: vi.fn(async () => true),
        ...overrides
      }
    }

    it('keeps a connection that answers the re-probe after the backoff', async () => {
      const probe = vi
        .fn()
        .mockRejectedValueOnce(new Error('ECONNRESET'))
        .mockRejectedValueOnce(new Error('ECONNRESET'))
        .mockRejectedValueOnce(new Error('ECONNRESET'))
        .mockResolvedValueOnce({ ok: true })

      const confirmDrop = confirmation()
      const test = harness({ confirmDrop, probe })

      await expect(revalidateRemoteConnection(test.options)).resolves.toEqual({ ok: true, rebuilt: false })
      await expect(revalidateRemoteConnection(test.options)).resolves.toEqual({ ok: true, rebuilt: false })
      await expect(revalidateRemoteConnection(test.options)).resolves.toEqual({ ok: true, rebuilt: false })

      expect(confirmDrop.backoff).toHaveBeenCalledOnce()
      expect(confirmDrop.transportAlive).toHaveBeenCalledOnce()
      expect(probe).toHaveBeenCalledTimes(4)
      expect(test.resetConnection).not.toHaveBeenCalled()
      expect(test.log).toHaveBeenLastCalledWith(expect.stringContaining('answered after'))
    })

    it('still drops a connection whose re-probe fails after the backoff', async () => {
      const probe = vi.fn().mockRejectedValue(new Error('ECONNRESET'))
      const confirmDrop = confirmation()
      const test = harness({ confirmDrop, probe })

      await revalidateRemoteConnection(test.options)
      await revalidateRemoteConnection(test.options)
      await expect(revalidateRemoteConnection(test.options)).resolves.toEqual({ ok: true, rebuilt: true })

      expect(confirmDrop.backoff).toHaveBeenCalledOnce()
      expect(probe).toHaveBeenCalledTimes(4)
      expect(test.resetConnection).toHaveBeenCalledOnce()
      expect(test.log).toHaveBeenLastCalledWith(expect.stringContaining('dropping stale connection'))
    })

    it('drops without a re-probe when the transport itself is gone', async () => {
      const probe = vi.fn().mockRejectedValue(new Error('ECONNRESET'))
      const confirmDrop = confirmation({ transportAlive: vi.fn(async () => false) })
      const test = harness({ confirmDrop, probe })

      await revalidateRemoteConnection(test.options)
      await revalidateRemoteConnection(test.options)
      await expect(revalidateRemoteConnection(test.options)).resolves.toEqual({ ok: true, rebuilt: true })

      // Three liveness probes and no fourth: a dead transport needs no re-probe.
      expect(probe).toHaveBeenCalledTimes(3)
      expect(test.resetConnection).toHaveBeenCalledOnce()
    })

    it('drops on the unchanged fast path when no confirmation is wired', async () => {
      const probe = vi.fn().mockRejectedValue(new Error('offline'))
      const test = harness({ probe })

      await revalidateRemoteConnection(test.options)
      await revalidateRemoteConnection(test.options)
      await expect(revalidateRemoteConnection(test.options)).resolves.toEqual({ ok: true, rebuilt: true })

      expect(probe).toHaveBeenCalledTimes(3)
      expect(test.resetConnection).toHaveBeenCalledOnce()
    })
  })
})

describe('ensureHealthyPooledRemoteBackendForDispatch', () => {
  it('retires a dead cached descriptor and gives dispatch the replacement', async () => {
    const stale = { baseUrl: 'http://127.0.0.1:49525', mode: 'remote' }
    const replacement = { baseUrl: 'http://127.0.0.1:53968', mode: 'remote' }
    const stalePromise = Promise.resolve(stale)
    let currentPromise: Promise<typeof stale> | null = stalePromise

    const retire = vi.fn(async () => {
      currentPromise = null
    })

    const reconnect = vi.fn(async () => {
      currentPromise = Promise.resolve(replacement)

      return replacement
    })

    const probe = vi.fn(async connection => {
      if (connection === stale) {
        throw new Error('connect ECONNREFUSED 127.0.0.1:49525')
      }
    })

    await expect(
      ensureHealthyPooledRemoteBackendForDispatch({
        connectionPromise: stalePromise,
        currentConnectionPromise: () => currentPromise,
        probe,
        reconnect,
        retire
      })
    ).resolves.toBe(replacement)

    expect(probe).toHaveBeenCalledWith(stale, '/api/status', {
      timeoutMs: POOLED_REMOTE_DISPATCH_PROBE_TIMEOUT_MS
    })
    expect(retire).toHaveBeenCalledOnce()
    expect(reconnect).toHaveBeenCalledOnce()
  })

  describe('host events', () => {
    // One busy host, many pooled profiles: the harness models a whole
    // connection so a probe failure can be correlated the way the real pool
    // sees it, instead of one descriptor in isolation.
    const hostHarness = () => {
      const tracker = new RemoteHostEventTracker()
      const state = { backendAnswers: true, backoffs: 0, hostReachable: true, probeFails: true }

      const probe = vi.fn(async () => {
        if (state.probeFails) {
          throw new Error('read ECONNRESET')
        }
      })

      const dispatch = (poolKey: string, connection: { baseUrl: string; mode: string }) => {
        const connectionPromise = Promise.resolve(connection)
        const retire = vi.fn(async () => {})
        const reconnect = vi.fn(async () => connection)

        const result = ensureHealthyPooledRemoteBackendForDispatch({
          connectionPromise,
          currentConnectionPromise: () => connectionPromise,
          hostEvent: {
            backoff: async () => {
              state.backoffs += 1
              // The backoff is where a busy host recovers; model that by
              // letting the descriptor answer again once it has elapsed.
              state.probeFails = !state.backendAnswers
            },
            classify: () => tracker.recordProbeFailure('conn:remote', poolKey),
            hostAlive: async () => state.hostReachable
          },
          probe,
          reconnect,
          retire
        })

        return { connection, reconnect, result, retire }
      }

      return { dispatch, probe, state, tracker }
    }

    const descriptorFor = (poolKey: string) => ({ baseUrl: `http://127.0.0.1/${poolKey}`, mode: 'remote' })

    it('defers teardown once distinct profiles fail together and keeps a descriptor that recovers', async () => {
      const host = hostHarness()
      const keys = ['conn:remote::alpha', 'conn:remote::beta', 'conn:remote::gamma']

      const dispatches = keys.map(key => host.dispatch(key, descriptorFor(key)))

      await Promise.all(dispatches.map(dispatch => dispatch.result))

      // The first two are below the distinct-profile threshold, so they take
      // the unchanged fast path; the third crosses it and must be spared.
      expect(dispatches[2].retire).not.toHaveBeenCalled()
      expect(dispatches[2].reconnect).not.toHaveBeenCalled()
      await expect(dispatches[2].result).resolves.toBe(dispatches[2].connection)
      expect(host.state.backoffs).toBe(1)
    })

    it('still retires a descriptor that is dead after the host-event backoff', async () => {
      const host = hostHarness()
      host.state.backendAnswers = false
      const keys = ['conn:remote::alpha', 'conn:remote::beta', 'conn:remote::gamma']

      const dispatches = keys.map(key => host.dispatch(key, descriptorFor(key)))

      await Promise.all(dispatches.map(dispatch => dispatch.result))

      expect(host.state.backoffs).toBe(1)
      expect(dispatches[2].retire).toHaveBeenCalledOnce()
      expect(dispatches[2].reconnect).toHaveBeenCalledOnce()
    })

    it('retires when the host itself is still unreachable after the backoff', async () => {
      const host = hostHarness()
      host.state.hostReachable = false
      host.state.backendAnswers = true
      const keys = ['conn:remote::alpha', 'conn:remote::beta', 'conn:remote::gamma']

      const dispatches = keys.map(key => host.dispatch(key, descriptorFor(key)))

      await Promise.all(dispatches.map(dispatch => dispatch.result))

      expect(dispatches[2].retire).toHaveBeenCalledOnce()
    })

    it('retires a lone failing profile immediately, with no backoff', async () => {
      const host = hostHarness()
      const solo = host.dispatch('conn:remote::alpha', descriptorFor('alpha'))

      await solo.result

      expect(host.state.backoffs).toBe(0)
      expect(solo.retire).toHaveBeenCalledOnce()
      expect(solo.reconnect).toHaveBeenCalledOnce()
    })
  })
})

describe('RemoteHostEventTracker', () => {
  it('classifies a host event only after enough DISTINCT profiles fail in the window', () => {
    const tracker = new RemoteHostEventTracker()

    expect(tracker.recordProbeFailure('conn:a', 'conn:a::one')).toBe(false)
    expect(tracker.recordProbeFailure('conn:a', 'conn:a::two')).toBe(false)
    expect(tracker.recordProbeFailure('conn:a', 'conn:a::three')).toBe(true)
    expect(tracker.recordProbeFailure('conn:a', 'conn:a::four')).toBe(true)
  })

  it('never classifies one profile failing repeatedly as a host event', () => {
    const tracker = new RemoteHostEventTracker()

    for (let attempt = 0; attempt < HOST_EVENT_DISTINCT_PROFILE_THRESHOLD + 2; attempt += 1) {
      expect(tracker.recordProbeFailure('conn:a', 'conn:a::one')).toBe(false)
    }
  })

  it('tracks connections independently', () => {
    const tracker = new RemoteHostEventTracker()

    tracker.recordProbeFailure('conn:a', 'conn:a::one')
    tracker.recordProbeFailure('conn:a', 'conn:a::two')

    expect(tracker.recordProbeFailure('conn:b', 'conn:b::one')).toBe(false)
    expect(tracker.recordProbeFailure('conn:a', 'conn:a::three')).toBe(true)
  })

  it('ages failures out of the window instead of latching', () => {
    let now = 0
    const tracker = new RemoteHostEventTracker(HOST_EVENT_DISTINCT_PROFILE_THRESHOLD, HOST_EVENT_WINDOW_MS, () => now)

    expect(tracker.recordProbeFailure('conn:a', 'conn:a::one')).toBe(false)
    expect(tracker.recordProbeFailure('conn:a', 'conn:a::two')).toBe(false)

    now += HOST_EVENT_WINDOW_MS + 1

    // The earlier two are stale, so an isolated later failure is not a herd.
    expect(tracker.recordProbeFailure('conn:a', 'conn:a::three')).toBe(false)
  })

  it('rejects a threshold that could fire on a single profile', () => {
    expect(() => new RemoteHostEventTracker(1)).toThrow(/at least 2/)
  })
})

describe('PooledRemoteDialGate', () => {
  const gateHarness = (limit = POOLED_REMOTE_DIAL_CONCURRENCY) => {
    const delays: number[] = []
    const releases: Array<() => void> = []
    let concurrent = 0
    let peak = 0

    const gate = new PooledRemoteDialGate({
      delay: async ms => {
        delays.push(ms)
      },
      jitterMs: POOLED_REMOTE_DIAL_JITTER_MS,
      limit,
      random: () => 0.5
    })

    const dial = (connectionId: string) => {
      const result = gate.run(connectionId, async () => {
        concurrent += 1
        peak = Math.max(peak, concurrent)

        await new Promise<void>(resolve => {
          releases.push(() => {
            concurrent -= 1
            resolve()
          })
        })

        return connectionId
      })

      return result
    }

    return {
      delays,
      dial,
      gate,
      peak: () => peak,
      releaseAll: async () => {
        while (releases.length > 0) {
          releases.shift()?.()
          // Let the released slot's waiter be admitted before the next round.
          await new Promise<void>(resolve => setTimeout(resolve, 0))
        }
      }
    }
  }

  it('never lets one connection exceed the dial concurrency cap', async () => {
    const harness = gateHarness()
    const dials = Array.from({ length: 21 }, () => harness.dial('conn:remote'))

    await new Promise<void>(resolve => setTimeout(resolve, 0))

    expect(harness.gate.active('conn:remote')).toBe(POOLED_REMOTE_DIAL_CONCURRENCY)

    await harness.releaseAll()
    await Promise.all(dials)

    expect(harness.peak()).toBe(POOLED_REMOTE_DIAL_CONCURRENCY)
  })

  it('jitters every queued admission but never the first ones', async () => {
    const harness = gateHarness()
    const dials = Array.from({ length: 5 }, () => harness.dial('conn:remote'))

    await new Promise<void>(resolve => setTimeout(resolve, 0))

    expect(harness.delays).toEqual([])

    await harness.releaseAll()
    await Promise.all(dials)

    // Two callers were queued, so exactly two admissions paid jitter.
    expect(harness.delays).toEqual([POOLED_REMOTE_DIAL_JITTER_MS / 2, POOLED_REMOTE_DIAL_JITTER_MS / 2])
  })

  it('does not make one connection wait behind another', async () => {
    const harness = gateHarness(1)
    const first = harness.dial('conn:one')
    const second = harness.dial('conn:two')

    await new Promise<void>(resolve => setTimeout(resolve, 0))

    expect(harness.peak()).toBe(2)

    await harness.releaseAll()
    await Promise.all([first, second])
  })

  it('frees the slot when a dial rejects', async () => {
    const gate = new PooledRemoteDialGate({ delay: async () => {}, limit: 1 })

    await expect(
      gate.run('conn:remote', () => {
        throw new Error('ssh: connect to host remote port 22: Connection refused')
      })
    ).rejects.toThrow('Connection refused')

    await expect(gate.run('conn:remote', async () => 'ok')).resolves.toBe('ok')
    expect(gate.active('conn:remote')).toBe(0)
  })
})

describe('revalidatePooledRemoteBackends', () => {
  interface TestRemoteConnection {
    authMode?: string
    baseUrl: string
    process?: unknown
    remoteBaseUrl?: null | string
  }

  const harness = (
    rawEntries: Array<[string, { process?: unknown; remoteBaseUrl?: null | string; authMode?: string }]>
  ) => {
    const entries: Array<[string, TestRemoteConnection & { connectionPromise: Promise<TestRemoteConnection> }]> =
      rawEntries.map(([profile, entry]) => {
        const connection = { ...entry, baseUrl: String(entry.remoteBaseUrl || '') }

        return [profile, { ...connection, connectionPromise: Promise.resolve(connection) }]
      })

    const unreachable = new Set<string>()
    const log = vi.fn()
    const stopBackend = vi.fn()

    const probe = vi.fn(async (connection: TestRemoteConnection) => {
      if ([...unreachable].some(base => connection.remoteBaseUrl?.startsWith(base))) {
        throw new Error('unreachable')
      }

      return {}
    })

    return {
      log,
      probe,
      stopBackend,
      unreachable,
      run: (tracker: RemoteLivenessTracker) =>
        revalidatePooledRemoteBackends({ entries, log, probe, stopBackend, tracker })
    }
  }

  it('probes only pooled entries backed by a remote host', async () => {
    const local = { process: {}, remoteBaseUrl: null }
    const spawning = { process: null, remoteBaseUrl: null }
    const remote = { process: null, remoteBaseUrl: 'https://remote.example.com' }

    const pool = harness([
      ['local', local],
      ['spawning', spawning],
      ['remote', remote]
    ])

    await pool.run(new RemoteLivenessTracker())

    expect(pool.probe).toHaveBeenCalledTimes(1)
    expect(pool.probe).toHaveBeenCalledWith(
      expect.objectContaining({ remoteBaseUrl: 'https://remote.example.com' }),
      '/api/status',
      { timeoutMs: REMOTE_LIVENESS_TIMEOUT_MS }
    )
    expect(pool.stopBackend).not.toHaveBeenCalled()
  })

  it('passes the authenticated OAuth descriptor to the liveness probe', async () => {
    const remote = {
      process: null,
      remoteBaseUrl: 'https://remote.example.com',
      authMode: 'oauth'
    }

    const pool = harness([['oauth', remote]])

    await pool.run(new RemoteLivenessTracker())

    expect(pool.probe).toHaveBeenCalledWith(expect.objectContaining({ authMode: 'oauth' }), '/api/status', {
      timeoutMs: REMOTE_LIVENESS_TIMEOUT_MS
    })
    expect(pool.stopBackend).not.toHaveBeenCalled()
  })

  it('drops a descriptor only after the shared failure limit', async () => {
    const pool = harness([['coder', { process: null, remoteBaseUrl: 'https://remote.example.com/' }]])
    pool.unreachable.add('https://remote.example.com')

    const tracker = new RemoteLivenessTracker()

    for (let attempt = 1; attempt < REMOTE_LIVENESS_FAILURE_LIMIT; attempt += 1) {
      await expect(pool.run(tracker)).resolves.toEqual({ dropped: [] })
      expect(pool.stopBackend).not.toHaveBeenCalled()
    }

    await expect(pool.run(tracker)).resolves.toEqual({ dropped: ['coder'] })
    expect(pool.stopBackend).toHaveBeenCalledWith('coder')
  })

  it('clears the streak when the host answers again', async () => {
    const pool = harness([['coder', { process: null, remoteBaseUrl: 'https://remote.example.com' }]])
    const tracker = new RemoteLivenessTracker()

    pool.unreachable.add('https://remote.example.com')
    await pool.run(tracker)

    pool.unreachable.clear()
    await pool.run(tracker)

    pool.unreachable.add('https://remote.example.com')

    for (let attempt = 1; attempt < REMOTE_LIVENESS_FAILURE_LIMIT; attempt += 1) {
      await expect(pool.run(tracker)).resolves.toEqual({ dropped: [] })
    }

    expect(pool.stopBackend).not.toHaveBeenCalled()
    await expect(pool.run(tracker)).resolves.toEqual({ dropped: ['coder'] })
  })

  it('keeps a healthy sibling when another profile on a different host dies', async () => {
    const pool = harness([
      ['coder', { process: null, remoteBaseUrl: 'https://dead.example.com' }],
      ['writer', { process: null, remoteBaseUrl: 'https://live.example.com' }]
    ])

    pool.unreachable.add('https://dead.example.com')

    const tracker = new RemoteLivenessTracker()

    for (let attempt = 1; attempt < REMOTE_LIVENESS_FAILURE_LIMIT; attempt += 1) {
      await pool.run(tracker)
    }

    await expect(pool.run(tracker)).resolves.toEqual({ dropped: ['coder'] })
    expect(pool.stopBackend).toHaveBeenCalledTimes(1)
    expect(pool.stopBackend).toHaveBeenCalledWith('coder')
  })
})
