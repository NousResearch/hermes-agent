import { describe, expect, it, vi } from 'vitest'

import {
  canReapSkewServingPid,
  listenPidsFromLsofT,
  localLoopbackListenPort,
  recycleOwnedBackend,
  recycleOwnedBackendTarget
} from './backend-recycle'

describe('canReapSkewServingPid', () => {
  it('reaps only a local hermes dashboard/serve pid, fail-closed otherwise', () => {
    expect(
      canReapSkewServingPid(4242, { command: 'hermes serve --port 9119', selfPid: 1 })
    ).toBe(true)
    expect(
      canReapSkewServingPid(4242, { command: 'python -m hermes_cli.main dashboard', selfPid: 1 })
    ).toBe(true)
    expect(canReapSkewServingPid(4242, { command: 'ssh', selfPid: 1 })).toBe(false)
    expect(canReapSkewServingPid(4242, { command: null, selfPid: 1 })).toBe(false)
    expect(
      canReapSkewServingPid(1, { command: 'hermes serve --port 9119', selfPid: 99 })
    ).toBe(false)
    expect(
      canReapSkewServingPid(99, { command: 'hermes serve --port 9119', selfPid: 99 })
    ).toBe(false)
  })
})

describe('localLoopbackListenPort', () => {
  it('returns the port only for loopback picker URLs', () => {
    expect(localLoopbackListenPort('http://127.0.0.1:9119')).toBe(9119)
    expect(localLoopbackListenPort('http://localhost:43210')).toBe(43210)
    expect(localLoopbackListenPort('http://example.com:9119')).toBeNull()
    expect(localLoopbackListenPort('')).toBeNull()
  })
})

describe('listenPidsFromLsofT', () => {
  it('parses unique positive pids from lsof -t output', () => {
    expect(listenPidsFromLsofT('4242\n4242\n99\n')).toEqual([4242, 99])
    expect(listenPidsFromLsofT('')).toEqual([])
  })
})

describe('recycleOwnedBackendTarget', () => {
  it('treats an empty or matching profile as the primary backend', () => {
    expect(recycleOwnedBackendTarget(undefined, 'default')).toBe('primary')
    expect(recycleOwnedBackendTarget('', 'default')).toBe('primary')
    expect(recycleOwnedBackendTarget('default', 'default')).toBe('primary')
  })

  it('treats any other named profile as a pooled backend', () => {
    expect(recycleOwnedBackendTarget('paid-ads', 'default')).toBe('pool')
  })
})

describe('recycleOwnedBackend', () => {
  it('kills the owned SSH serve before the primary child, then notifies apply', async () => {
    const events: string[] = []

    const target = await recycleOwnedBackend({
      notifyApplied: () => events.push('applied'),
      primaryProfile: 'default',
      profile: undefined,
      teardownPool: async () => {
        events.push('pool')
      },
      teardownPrimary: async () => {
        events.push('primary')
      },
      teardownSsh: async profile => {
        events.push(`ssh:${profile}`)
      }
    })

    expect(target).toBe('primary')
    expect(events).toEqual(['ssh:', 'primary', 'applied'])
  })

  it('recycles a pooled profile without tearing down the primary', async () => {
    const events: string[] = []

    const target = await recycleOwnedBackend({
      notifyApplied: () => events.push('applied'),
      primaryProfile: 'default',
      profile: 'paid-ads',
      teardownPool: async profile => {
        events.push(`pool:${profile}`)
      },
      teardownPrimary: async () => {
        events.push('primary')
      },
      teardownSsh: async profile => {
        events.push(`ssh:${profile}`)
      }
    })

    expect(target).toBe('pool')
    expect(events).toEqual(['ssh:paid-ads', 'pool:paid-ads'])
  })

  it('awaits SSH teardown before the local child even when SSH is slow', async () => {
    const events: string[] = []
    let releaseSsh!: () => void

    const sshGate = new Promise<void>(resolve => {
      releaseSsh = resolve
    })

    const run = recycleOwnedBackend({
      notifyApplied: () => events.push('applied'),
      primaryProfile: 'default',
      teardownPool: vi.fn(),
      teardownPrimary: async () => {
        events.push('primary')
      },
      teardownSsh: async () => {
        events.push('ssh-start')
        await sshGate
        events.push('ssh-done')
      }
    })

    await Promise.resolve()
    expect(events).toEqual(['ssh-start'])

    releaseSsh()
    await run

    expect(events).toEqual(['ssh-start', 'ssh-done', 'primary', 'applied'])
  })

  it('reaps the leftover 503-serving pid, not only the Electron-owned child (#101561)', async () => {
    const events: string[] = []
    const teardownServingPid = vi.fn(async (pid: number) => {
      events.push(`serving:${pid}`)
    })

    const target = await recycleOwnedBackend({
      notifyApplied: () => events.push('applied'),
      primaryProfile: 'default',
      servingPid: 4242,
      teardownPool: async () => {
        events.push('pool')
      },
      teardownPrimary: async () => {
        events.push('primary')
      },
      teardownServingPid,
      teardownSsh: async profile => {
        events.push(`ssh:${profile}`)
      }
    })

    expect(target).toBe('primary')
    expect(teardownServingPid).toHaveBeenCalledWith(4242)
    expect(events).toEqual(['ssh:', 'serving:4242', 'primary', 'applied'])
  })

  it('still recycles the owned child if leftover teardown throws', async () => {
    const events: string[] = []

    const target = await recycleOwnedBackend({
      notifyApplied: () => events.push('applied'),
      primaryProfile: 'default',
      servingPid: 4242,
      teardownPool: vi.fn(),
      teardownPrimary: async () => {
        events.push('primary')
      },
      teardownServingPid: async () => {
        events.push('serving-throw')
        throw new Error('leftover refused')
      },
      teardownSsh: async () => {
        events.push('ssh')
      }
    })

    expect(target).toBe('primary')
    expect(events).toEqual(['ssh', 'serving-throw', 'primary', 'applied'])
  })

  it('reaps the loopback listener when the leftover 503 has no pid (#101561)', async () => {
    const events: string[] = []
    const teardownListenPort = vi.fn(async (port: number) => {
      events.push(`listen:${port}`)
    })
    const teardownServingPid = vi.fn()

    const target = await recycleOwnedBackend({
      listenPort: 9119,
      notifyApplied: () => events.push('applied'),
      primaryProfile: 'default',
      teardownListenPort,
      teardownPool: vi.fn(),
      teardownPrimary: async () => {
        events.push('primary')
      },
      teardownServingPid,
      teardownSsh: async () => {
        events.push('ssh')
      }
    })

    expect(target).toBe('primary')
    expect(teardownServingPid).not.toHaveBeenCalled()
    expect(teardownListenPort).toHaveBeenCalledWith(9119)
    expect(events).toEqual(['ssh', 'listen:9119', 'primary', 'applied'])
  })
})
