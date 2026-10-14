import { describe, expect, it, vi } from 'vitest'

import { recycleOwnedBackend, recycleOwnedBackendTarget } from './backend-recycle'

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
})
