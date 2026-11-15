import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { setApiRequestConnection, setApiRequestProfile } from '@/api/client'
import type { DesktopVersionInfo } from '@/global'

const storage = new Map<string, string>()

vi.mock('@/lib/storage', () => ({
  persistBoolean: (key: string, value: boolean) => {
    storage.set(key, String(value))
  },
  persistString: (key: string, value: null | string) => {
    if (value === null) {
      storage.delete(key)
    } else {
      storage.set(key, value)
    }
  },
  // store/session persists its exact owner hints through the JSON helpers.
  readJson: (key: string) => {
    const value = storage.get(key)

    try {
      return value === undefined ? null : JSON.parse(value)
    } catch {
      return null
    }
  },
  writeJson: (key: string, value: unknown) => {
    if (value === null) {
      storage.delete(key)
    } else {
      storage.set(key, JSON.stringify(value))
    }
  },
  storedBoolean: (key: string, fallback: boolean) => {
    const value = storage.get(key)

    return value === undefined ? fallback : value === 'true'
  },
  storedString: (key: string) => storage.get(key) ?? null
}))

// Import-safety stubs: updates.ts pulls these modules in at load time.
vi.mock('@/store/notifications', () => ({ notify: vi.fn(), dismissNotification: vi.fn() }))

// The real connections store drags in the profile/gateway-switch graph.
const { atom: registryAtom } = await import('nanostores')
const $mockConnectionsRegistry = registryAtom<unknown>(null)

vi.mock('@/store/connections', () => ({
  $connectionsRegistry: $mockConnectionsRegistry,
  refreshConnectionsRegistry: () => Promise.resolve($mockConnectionsRegistry.get())
}))

vi.mock('@/hermes', () => ({ checkHermesUpdate: vi.fn(), updateHermes: vi.fn(), getActionStatus: vi.fn() }))
vi.mock('@/store/gateway-reconnect', () => ({ reconnectGateway: vi.fn() }))

const { refreshDesktopVersion, $desktopVersion, startUpdatePoller, stopUpdatePoller } = await import('./updates')

const { setConnection } = await import('./session')

const setRemote = (on: boolean) =>
  setConnection({
    baseUrl: 'http://box:9119',
    isFullscreen: false,
    mode: on ? 'remote' : 'local',
    nativeOverlayWidth: 0,
    token: 't',
    wsUrl: 'ws://box:9119',
    logs: [],
    windowButtonPosition: null
  })

describe('desktop version follows the request scope', () => {
  let getVersion: ReturnType<typeof vi.fn>

  const info = (appVersion: string): DesktopVersionInfo =>
    ({ appVersion, electronVersion: '40', nodeVersion: '26', platform: 'linux', hermesRoot: '' }) as DesktopVersionInfo

  beforeEach(() => {
    setRemote(false)
    setApiRequestConnection(null)
    setApiRequestProfile(null)
    $desktopVersion.set(null)
    getVersion = vi.fn().mockResolvedValue(info('1.0.0'))
    // Same minimal window the startUpdatePoller suite installs (siblings delete it).
    ;(globalThis as unknown as { window: unknown }).window = {
      hermesDesktop: {
        getVersion,
        updates: { check: vi.fn().mockResolvedValue({ supported: false }), onProgress: vi.fn(() => () => {}) }
      },
      addEventListener: vi.fn(),
      removeEventListener: vi.fn()
    }
    stopUpdatePoller()
  })

  afterEach(() => {
    stopUpdatePoller()
    delete (globalThis as unknown as { window?: unknown }).window
    setApiRequestConnection(null)
    setApiRequestProfile(null)
    $desktopVersion.set(null)
  })

  it('re-fetches when switching from a remote connection back to local', async () => {
    setRemote(true)
    setApiRequestConnection('remote-box')
    startUpdatePoller()
    await vi.waitFor(() => expect(getVersion).toHaveBeenCalled())
    getVersion.mockClear()
    getVersion.mockResolvedValue(info('0.22.1+72'))

    setApiRequestConnection('local')
    setRemote(false)

    await vi.waitFor(() => expect(getVersion).toHaveBeenCalledWith(expect.objectContaining({ connectionId: 'local' })))
    await vi.waitFor(() => expect($desktopVersion.get()?.appVersion).toBe('0.22.1+72'))
  })

  it('re-fetches when only the profile changes on the same connection', async () => {
    setApiRequestConnection('remote-box')
    setRemote(true)
    startUpdatePoller()
    await vi.waitFor(() => expect(getVersion).toHaveBeenCalled())
    getVersion.mockClear()

    setApiRequestProfile('work')

    await vi.waitFor(() =>
      expect(getVersion).toHaveBeenCalledWith(expect.objectContaining({ connectionId: 'remote-box', profile: 'work' }))
    )
  })

  it('keeps the newest reply when two same-scope requests resolve out of order', async () => {
    setApiRequestConnection('remote-box')
    setRemote(true)

    let releaseOld: (value: DesktopVersionInfo) => void = () => {}
    getVersion.mockReset()
    getVersion
      .mockImplementationOnce(() => new Promise<DesktopVersionInfo>(resolve => (releaseOld = resolve)))
      .mockResolvedValueOnce(info('newer'))

    const older = refreshDesktopVersion()
    expect(await refreshDesktopVersion()).toEqual(info('newer'))

    releaseOld(info('older'))
    expect(await older).toBeNull()
    expect($desktopVersion.get()?.appVersion).toBe('newer')
  })

  it('drops a reply whose scope changed in flight even when nothing re-requests', async () => {
    setApiRequestConnection('remote-box')
    setRemote(true)

    let release: (value: DesktopVersionInfo) => void = () => {}
    getVersion.mockReset()
    getVersion.mockImplementationOnce(() => new Promise<DesktopVersionInfo>(resolve => (release = resolve)))

    const pending = refreshDesktopVersion()
    setApiRequestProfile('work')
    release(info('old-profile'))

    expect(await pending).toBeNull()
    expect($desktopVersion.get()).toBeNull()
  })

  it('ignores a slower reply from a request superseded by a profile switch', async () => {
    setApiRequestConnection('remote-box')
    setRemote(true)
    startUpdatePoller()
    await vi.waitFor(() => expect(getVersion).toHaveBeenCalled())

    let releaseOld: (value: DesktopVersionInfo) => void = () => {}
    getVersion.mockReset()
    getVersion
      .mockImplementationOnce(() => new Promise<DesktopVersionInfo>(resolve => (releaseOld = resolve)))
      .mockResolvedValue(info('new-profile'))

    const stale = refreshDesktopVersion()
    setApiRequestProfile('work')
    await vi.waitFor(() => expect($desktopVersion.get()?.appVersion).toBe('new-profile'))

    releaseOld(info('old-profile'))
    expect(await stale).toBeNull()
    expect($desktopVersion.get()?.appVersion).toBe('new-profile')
  })
})
