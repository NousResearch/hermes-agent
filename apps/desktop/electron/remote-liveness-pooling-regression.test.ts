import { describe, expect, it, vi } from 'vitest'

import {
  REMOTE_LIVENESS_FAILURE_LIMIT,
  RemoteLivenessTracker,
  revalidatePooledRemoteBackends,
  revalidateSuspectPooledRemoteBackends
} from './remote-liveness'

const remote = (baseUrl: string, profile: string, sharedProbeKey?: string) => ({
  connectionPromise: Promise.resolve({ baseUrl, profile }),
  process: null,
  remoteBaseUrl: baseUrl,
  sharedProbeKey
})

describe('pooled remote liveness ownership', () => {
  it('coalesces profiles owned by one registry connection', async () => {
    const entries: Array<[string, ReturnType<typeof remote>]> = [
      ['conn:source-a::coder', remote('https://remote.example.com', 'coder')],
      ['conn:source-a::writer', remote('https://remote.example.com/', 'writer')]
    ]
    const probe = vi.fn(async () => {
      throw new Error('offline')
    })
    const stopBackend = vi.fn()
    const tracker = new RemoteLivenessTracker()

    for (let attempt = 1; attempt < REMOTE_LIVENESS_FAILURE_LIMIT; attempt += 1) {
      await expect(
        revalidatePooledRemoteBackends({ entries, log: vi.fn(), probe, stopBackend, tracker })
      ).resolves.toEqual({ dropped: [] })
    }

    await expect(
      revalidatePooledRemoteBackends({ entries, log: vi.fn(), probe, stopBackend, tracker })
    ).resolves.toEqual({ dropped: ['conn:source-a::coder', 'conn:source-a::writer'] })
    expect(probe).toHaveBeenCalledTimes(REMOTE_LIVENESS_FAILURE_LIMIT)
    expect(stopBackend).toHaveBeenCalledTimes(2)
  })

  it('keeps unscoped same-URL descriptors independent', async () => {
    const entries: Array<[string, ReturnType<typeof remote>]> = [
      ['dead', remote('https://remote.example.com', 'dead')],
      ['healthy', remote('https://remote.example.com', 'healthy')]
    ]
    const probe = vi.fn(async connection => {
      if (connection.profile === 'dead') {
        throw new Error('offline')
      }
    })
    const stopBackend = vi.fn()

    await expect(
      revalidatePooledRemoteBackends({
        entries,
        log: vi.fn(),
        probe,
        stopBackend,
        tracker: new RemoteLivenessTracker(1)
      })
    ).resolves.toEqual({ dropped: ['dead'] })
    expect(probe).toHaveBeenCalledTimes(2)
    expect(stopBackend).toHaveBeenCalledOnce()
    expect(stopBackend).toHaveBeenCalledWith('dead')
  })

  it('uses the same ownership key when a post-resume rebuild clears failures', async () => {
    const baseUrl = 'https://remote.example.com'
    const recovering = remote(baseUrl, 'recovering', 'source-a')
    const unrelated = remote(baseUrl, 'unrelated', 'source-b')
    const tracker = new RemoteLivenessTracker(3, 60_000, () => 1_000)
    const offline = vi.fn(async () => {
      throw new Error('offline')
    })

    const sweep = (entries: Array<[string, ReturnType<typeof remote>]>) =>
      revalidatePooledRemoteBackends({ entries, log: vi.fn(), probe: offline, stopBackend: vi.fn(), tracker })

    await sweep([
      ['recovering', recovering],
      ['unrelated', unrelated]
    ])
    await sweep([
      ['recovering', recovering],
      ['unrelated', unrelated]
    ])

    await expect(
      revalidateSuspectPooledRemoteBackends({
        entries: [['recovering', recovering]],
        log: vi.fn(),
        probe: offline,
        rebuild: vi.fn(async () => undefined),
        retire: vi.fn(async () => undefined),
        tracker
      })
    ).resolves.toEqual({ rebuilt: ['recovering'], retired: ['recovering'] })

    await expect(
      sweep([
        ['recovering', recovering],
        ['unrelated', unrelated]
      ])
    ).resolves.toEqual({ dropped: ['unrelated'] })
  })
})
