import { describe, expect, it, vi } from 'vitest'

import {
  ensureHealthyPooledRemoteBackendForDispatch,
  POOLED_REMOTE_DISPATCH_PROBE_TIMEOUT_MS,
  REMOTE_LIVENESS_FAILURE_LIMIT,
  REMOTE_LIVENESS_FAILURE_WINDOW_MS,
  REMOTE_LIVENESS_TIMEOUT_MS,
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
})

describe('ensureHealthyPooledRemoteBackendForDispatch', () => {
  it('retires a dead cached descriptor and gives dispatch the replacement on transport failure', async () => {
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

    expect(probe).toHaveBeenCalledWith(stale, '/api/health', {
      timeoutMs: POOLED_REMOTE_DISPATCH_PROBE_TIMEOUT_MS
    })
    expect(retire).toHaveBeenCalledOnce()
    expect(reconnect).toHaveBeenCalledOnce()
  })

  it('preserves the descriptor on a single transient timeout, returning existing connection', async () => {
    const active = { baseUrl: 'http://127.0.0.1:49525', mode: 'remote' }
    const activePromise = Promise.resolve(active)

    const retire = vi.fn()
    const reconnect = vi.fn()

    const probe = vi.fn(async () => {
      throw new Error('Timed out connecting to Hermes backend after 10000ms')
    })

    await expect(
      ensureHealthyPooledRemoteBackendForDispatch({
        connectionPromise: activePromise,
        currentConnectionPromise: () => activePromise,
        probe,
        reconnect,
        retire
      })
    ).resolves.toBe(active)

    expect(retire).not.toHaveBeenCalled()
    expect(reconnect).not.toHaveBeenCalled()
  })

  it('retires and reconnects when timeouts repeat consecutively on a wedged backend', async () => {
    const wedged = { baseUrl: 'http://127.0.0.1:49525', mode: 'remote' }
    const replacement = { baseUrl: 'http://127.0.0.1:53968', mode: 'remote' }
    const wedgedPromise = Promise.resolve(wedged)
    let currentPromise: Promise<typeof wedged> | null = wedgedPromise

    const retire = vi.fn(async () => {
      currentPromise = null
    })
    const reconnect = vi.fn(async () => {
      currentPromise = Promise.resolve(replacement)

      return replacement
    })

    const probe = vi.fn(async () => {
      throw new Error('Timed out connecting to Hermes backend after 10000ms')
    })

    // First timeout: transient miss, preserved without retire
    await expect(
      ensureHealthyPooledRemoteBackendForDispatch({
        connectionPromise: wedgedPromise,
        currentConnectionPromise: () => currentPromise,
        probe,
        reconnect,
        retire
      })
    ).resolves.toBe(wedged)

    expect(retire).not.toHaveBeenCalled()
    expect(reconnect).not.toHaveBeenCalled()

    // Second consecutive timeout: repeated timeout exceeds tolerance, retires and reconnects
    await expect(
      ensureHealthyPooledRemoteBackendForDispatch({
        connectionPromise: wedgedPromise,
        currentConnectionPromise: () => currentPromise,
        probe,
        reconnect,
        retire
      })
    ).resolves.toBe(replacement)

    expect(retire).toHaveBeenCalledOnce()
    expect(reconnect).toHaveBeenCalledOnce()
  })

  it('resets consecutive timeout count after a successful probe', async () => {
    const flaky = { baseUrl: 'http://127.0.0.1:49525', mode: 'remote' }
    const flakyPromise = Promise.resolve(flaky)

    const retire = vi.fn()
    const reconnect = vi.fn()

    let shouldTimeout = true
    const probe = vi.fn(async () => {
      if (shouldTimeout) {
        throw new Error('Timed out connecting to Hermes backend after 10000ms')
      }
    })

    // 1st dispatch: times out (count = 1) -> tolerated
    await expect(
      ensureHealthyPooledRemoteBackendForDispatch({
        connectionPromise: flakyPromise,
        currentConnectionPromise: () => flakyPromise,
        probe,
        reconnect,
        retire
      })
    ).resolves.toBe(flaky)
    expect(retire).not.toHaveBeenCalled()

    // 2nd dispatch: succeeds -> resets count
    shouldTimeout = false
    await expect(
      ensureHealthyPooledRemoteBackendForDispatch({
        connectionPromise: flakyPromise,
        currentConnectionPromise: () => flakyPromise,
        probe,
        reconnect,
        retire
      })
    ).resolves.toBe(flaky)
    expect(retire).not.toHaveBeenCalled()

    // 3rd dispatch: times out again (count = 1 again, since reset) -> still tolerated
    shouldTimeout = true
    await expect(
      ensureHealthyPooledRemoteBackendForDispatch({
        connectionPromise: flakyPromise,
        currentConnectionPromise: () => flakyPromise,
        probe,
        reconnect,
        retire
      })
    ).resolves.toBe(flaky)
    expect(retire).not.toHaveBeenCalled()
  })

  it('falls back to /api/status when /api/health returns 404 on older backends', async () => {
    const legacy = { baseUrl: 'http://127.0.0.1:49525', mode: 'remote' }
    const legacyPromise = Promise.resolve(legacy)

    const retire = vi.fn()
    const reconnect = vi.fn()

    const probe = vi.fn(async (_connection, path) => {
      if (path === '/api/health') {
        throw new Error('404: Not Found')
      }
      // /api/status succeeds
    })

    await expect(
      ensureHealthyPooledRemoteBackendForDispatch({
        connectionPromise: legacyPromise,
        currentConnectionPromise: () => legacyPromise,
        probe,
        reconnect,
        retire
      })
    ).resolves.toBe(legacy)

    expect(probe).toHaveBeenCalledWith(legacy, '/api/health', {
      timeoutMs: POOLED_REMOTE_DISPATCH_PROBE_TIMEOUT_MS
    })
    expect(probe).toHaveBeenCalledWith(legacy, '/api/status', {
      timeoutMs: POOLED_REMOTE_DISPATCH_PROBE_TIMEOUT_MS
    })
    expect(retire).not.toHaveBeenCalled()
    expect(reconnect).not.toHaveBeenCalled()
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
