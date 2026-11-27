import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { backendTransitionFor, resetBackendTransitionsForTests } from '@/store/backend-transition'
import { $activeGatewayProfile, $profiles } from '@/store/profile'
import { $profileRemoteOverrides } from '@/store/profile-remote-override'
import { SWITCH_HOOK_MAX_MS, SWITCH_SETTLE_MAX_MS } from '@/store/switch-transition'

import { clearProfileBackend, profileBackendFeatures, setProfileBackend } from './profile-backend'

type Bridge = Record<string, ReturnType<typeof vi.fn>>

function installBridge(overrides: Bridge = {}): Bridge {
  const bridge: Bridge = {
    clearProfileBackend: vi.fn(async () => ({ applied: true })),
    confirmClearProfileBackend: vi.fn(async () => ({ confirmed: true, grant: 'g-clear' })),
    confirmSetProfileBackend: vi.fn(async () => ({ confirmed: true, grant: 'g-set' })),
    getConnectionConfig: vi.fn(async (name: string) =>
      name === 'work'
        ? { launcher: { plugin: 'acme' }, mode: 'remote', remoteUrl: 'http://127.0.0.1:47911' }
        : { launcher: null, mode: 'local', remoteUrl: '' }
    ),
    setProfileBackend: vi.fn(async () => ({ applied: true })),
    ...overrides
  }

  ;(window as unknown as { hermesDesktop: unknown }).hermesDesktop = bridge

  return bridge
}

beforeEach(() => {
  $profiles.set([
    { name: 'default', path: '/h' },
    { name: 'work', path: '/h/profiles/work' }
  ] as never)
  // The window shows another profile, so a switch's state ends with the call.
  $activeGatewayProfile.set('default')
})

afterEach(() => {
  resetBackendTransitionsForTests()
  $profileRemoteOverrides.set({})
  delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
})

describe('host.setProfileBackend', () => {
  it('routes a running backend through main and refreshes the rail badges', async () => {
    const bridge = installBridge()

    await expect(
      setProfileBackend({ plugin: ' acme ', profile: ' work ', token: 'tok', url: ' http://127.0.0.1:47911 ' })
    ).resolves.toEqual({ applied: true })

    expect(bridge.confirmSetProfileBackend).not.toHaveBeenCalled()
    expect(bridge.setProfileBackend).toHaveBeenCalledWith({
      plugin: 'acme',
      profile: 'work',
      token: 'tok',
      url: 'http://127.0.0.1:47911'
    })
    expect($profileRemoteOverrides.get().work).toEqual({
      host: '127.0.0.1:47911',
      launcher: { plugin: 'acme' },
      url: 'http://127.0.0.1:47911'
    })
    expect($profileRemoteOverrides.get().default).toBeUndefined()
  })

  it('runs prepare only after the user agrees, showing "starting" meanwhile, and sets with the grant', async () => {
    const order: string[] = []

    const bridge = installBridge({
      confirmSetProfileBackend: vi.fn(async () => {
        order.push('confirm')

        return { confirmed: true, grant: 'g-set' }
      }),
      setProfileBackend: vi.fn(async () => {
        order.push('set')

        return { applied: true }
      })
    })

    const prepare = vi.fn(async () => {
      order.push(`prepare:${backendTransitionFor('work')?.kind}`)

      return { token: 'tok', url: ' http://127.0.0.1:47972 ' }
    })

    await expect(
      setProfileBackend({ plugin: 'acme', prepare, profile: 'work', purpose: ' Runs work in a sandbox. ' })
    ).resolves.toEqual({ applied: true })

    expect(order).toEqual(['confirm', 'prepare:starting', 'set'])
    expect(backendTransitionFor('work')).toBeNull()
    expect(bridge.confirmSetProfileBackend).toHaveBeenCalledWith({
      plugin: 'acme',
      profile: 'work',
      purpose: 'Runs work in a sandbox.'
    })
    expect(bridge.setProfileBackend).toHaveBeenCalledWith({
      grant: 'g-set',
      plugin: 'acme',
      profile: 'work',
      purpose: 'Runs work in a sandbox.',
      token: 'tok',
      url: 'http://127.0.0.1:47972'
    })
  })

  it('a declined confirm never runs prepare; a failed prepare sets nothing', async () => {
    const bridge = installBridge({ confirmSetProfileBackend: vi.fn(async () => ({ confirmed: false })) })
    const prepare = vi.fn(async () => ({ token: 't', url: 'http://127.0.0.1:1' }))

    await expect(setProfileBackend({ plugin: 'acme', prepare, profile: 'work' })).resolves.toEqual({ applied: false })
    expect(prepare).not.toHaveBeenCalled()

    bridge.confirmSetProfileBackend.mockResolvedValue({ confirmed: true, grant: 'g' })
    prepare.mockRejectedValue(new Error('the sandbox did not start'))

    await expect(setProfileBackend({ plugin: 'acme', prepare, profile: 'work' })).rejects.toThrow(
      'the sandbox did not start'
    )
    expect(bridge.setProfileBackend).not.toHaveBeenCalled()
    expect(backendTransitionFor('work')).toBeNull()
  })

  it('rejects on a build without the bridge and names what this build supports', async () => {
    await expect(
      setProfileBackend({ plugin: 'acme', profile: 'work', token: 't', url: 'http://127.0.0.1:1' })
    ).rejects.toThrow('This build of Hermes Desktop cannot run a profile through a plugin.')
    expect(profileBackendFeatures).toEqual({ beforeApply: true, prepare: true, purpose: true })
  })
})

describe('host.clearProfileBackend', () => {
  it('runs beforeApply after the confirm and before the route changes, showing "stopping" meanwhile', async () => {
    const order: string[] = []

    const bridge = installBridge({
      clearProfileBackend: vi.fn(async () => {
        order.push('clear')

        return { applied: true }
      }),
      confirmClearProfileBackend: vi.fn(async () => {
        order.push('confirm')

        return { confirmed: true, grant: 'g-1' }
      })
    })

    await expect(
      clearProfileBackend({
        beforeApply: async () => void order.push(`stop:${backendTransitionFor('work')?.kind}`),
        plugin: 'acme',
        profile: 'work'
      })
    ).resolves.toEqual({ applied: true })

    expect(order).toEqual(['confirm', 'stop:stopping', 'clear'])
    expect(bridge.clearProfileBackend).toHaveBeenCalledWith({ grant: 'g-1', plugin: 'acme', profile: 'work' })
    expect(backendTransitionFor('work')).toBeNull()
  })

  it('without beforeApply main asks in the same call; a decline or a failed beforeApply changes nothing', async () => {
    const bridge = installBridge({ clearProfileBackend: vi.fn(async () => ({ applied: false })) })

    await expect(clearProfileBackend({ plugin: 'acme', profile: 'work' })).resolves.toEqual({ applied: false })
    expect(bridge.confirmClearProfileBackend).not.toHaveBeenCalled()
    expect(bridge.clearProfileBackend).toHaveBeenCalledWith({ plugin: 'acme', profile: 'work' })

    bridge.clearProfileBackend.mockClear()
    bridge.confirmClearProfileBackend.mockResolvedValueOnce({ confirmed: false })
    const beforeApply = vi.fn()

    await expect(clearProfileBackend({ beforeApply, plugin: 'acme', profile: 'work' })).resolves.toEqual({
      applied: false
    })
    expect(beforeApply).not.toHaveBeenCalled()

    await expect(
      clearProfileBackend({
        beforeApply: async () => {
          throw new Error('the sandbox did not stop')
        },
        plugin: 'acme',
        profile: 'work'
      })
    ).rejects.toThrow('the sandbox did not stop')
    expect(bridge.clearProfileBackend).not.toHaveBeenCalled()
    expect(backendTransitionFor('work')).toBeNull()
  })
})

// The plugin's own hook decides how long a switch takes (a first sandbox start can
// take minutes). While it runs, the profile is still on the backend the switch is about to retire.
describe('a switch whose plugin hook runs longer than three minutes', () => {
  afterEach(() => vi.useRealTimers())

  it('keeps "starting" for as long as prepare runs, so the composer holds messages until the switch', async () => {
    vi.useFakeTimers()
    installBridge()

    let started!: (address: { token: string; url: string }) => void
    const prepare = () => new Promise<{ token: string; url: string }>(resolve => (started = resolve))
    const switching = setProfileBackend({ plugin: 'acme', prepare, profile: 'work' })

    await vi.advanceTimersByTimeAsync(1_000)
    expect(backendTransitionFor('work')?.kind).toBe('starting')

    // 200 s in: the plugin is still starting its backend (the first user's own limit is 300 s).
    await vi.advanceTimersByTimeAsync(199_000)
    expect(backendTransitionFor('work')?.kind).toBe('starting')

    started({ token: 'tok', url: 'http://127.0.0.1:47972' })
    await expect(switching).resolves.toEqual({ applied: true })
  })

  it('keeps "stopping" for as long as beforeApply runs', async () => {
    vi.useFakeTimers()
    installBridge()

    let stopped!: () => void
    const beforeApply = () => new Promise<void>(resolve => (stopped = resolve))
    const switching = clearProfileBackend({ beforeApply, plugin: 'acme', profile: 'work' })

    await vi.advanceTimersByTimeAsync(200_000)
    expect(backendTransitionFor('work')?.kind).toBe('stopping')

    stopped()
    await expect(switching).resolves.toEqual({ applied: true })
  })

  it('a hook that never returns stops holding the composer when its time is up', async () => {
    vi.useFakeTimers()
    installBridge()
    void setProfileBackend({ plugin: 'acme', prepare: () => new Promise(() => undefined), profile: 'work' })

    await vi.advanceTimersByTimeAsync(SWITCH_HOOK_MAX_MS)
    expect(backendTransitionFor('work')?.kind).toBe('starting')

    await vi.advanceTimersByTimeAsync(SWITCH_SETTLE_MAX_MS)
    expect(backendTransitionFor('work')).toBeNull()
  })
})
