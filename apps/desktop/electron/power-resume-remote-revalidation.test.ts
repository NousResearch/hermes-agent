import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

import { describe, expect, it, vi } from 'vitest'

import {
  attachPowerResumeRemoteRevalidation,
  POWER_RESUME_REVALIDATION_HOLDOFF_MS,
  RemoteLivenessTracker,
  revalidateSuspectPooledRemoteBackends
} from './remote-liveness'

const here = path.dirname(fileURLToPath(import.meta.url))
const mainSource = fs.readFileSync(path.join(here, 'main.ts'), 'utf8').replace(/\r\n/g, '\n')

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

describe('main.ts wiring for #93910', () => {
  it('registers the suspect-pool revalidation on powerMonitor resume/unlock', () => {
    const fnStart = mainSource.indexOf('function registerPowerResumeListeners()')
    expect(fnStart).toBeGreaterThan(-1)
    const body = mainSource.slice(fnStart, mainSource.indexOf('\nfunction ', fnStart + 1))

    expect(body).toContain('attachPowerResumeRemoteRevalidation(')
    expect(body).toContain('revalidateSuspectPoolAfterResume()')
  })

  it('drives suspect revalidation through the shared coordinator, teardown and claimed re-dial primitives', () => {
    const fnStart = mainSource.indexOf('function revalidateSuspectPoolAfterResume()')
    expect(fnStart).toBeGreaterThan(-1)
    const body = mainSource.slice(fnStart, fnStart + 2_500)

    expect(body).toContain('remoteRevalidation.run(')
    expect(body).toContain('revalidateSuspectPooledRemoteBackends({')
    expect(body).toContain('stopPoolBackend(')
    expect(body).toContain('sshBootstrapCoordinator.cancelAndWait(')
    expect(body).toContain('teardownSshConnection(')
    expect(body).toContain('redialPoolBackendAfterResume')
    expect(body).toContain('tracker: remoteLiveness')
  })

  it('re-dials a retired pool key through the single-owner dial claim', () => {
    const fnStart = mainSource.indexOf('function redialPoolBackendAfterResume(')
    expect(fnStart).toBeGreaterThan(-1)
    const body = mainSource.slice(fnStart, fnStart + 1_200)

    expect(body).toContain('parseBackendScopeKey(')
    expect(body).toContain('backendDialClaims.run(')
    expect(body).toContain('ensureRegistryBackend(')
  })
})
