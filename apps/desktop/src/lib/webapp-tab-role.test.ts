import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import {
  _resetWebappTabRoleForTests,
  browserTabNavigationRole,
  webappTabReplaysRememberedNavigation
} from './webapp-tab-role'

const { browserHostMock } = vi.hoisted(() => ({ browserHostMock: vi.fn(() => false) }))

vi.mock('@/lib/platform', () => ({ isBrowserHostedDesktop: () => browserHostMock() }))

/** The Web Locks subset the election uses: `ifAvailable` answers null when taken. */
function installFakeLocks(held = new Set<string>()) {
  const request = vi.fn((name: string, options: { ifAvailable?: boolean }, callback: (lock: unknown) => unknown) => {
    if (options?.ifAvailable && held.has(name)) {
      return Promise.resolve(callback(null))
    }

    held.add(name)

    return Promise.resolve(callback({ name }))
  })

  Object.defineProperty(navigator, 'locks', { configurable: true, value: { request } })

  return { held, request }
}

describe('webapp tab-role election', () => {
  beforeEach(() => {
    _resetWebappTabRoleForTests()
    browserHostMock.mockReturnValue(false)
  })

  afterEach(() => {
    Reflect.deleteProperty(navigator, 'locks')
    vi.restoreAllMocks()
  })

  it('a native window owns its restore without any election', async () => {
    const { request } = installFakeLocks()

    expect(browserTabNavigationRole()).toBe(true)
    expect(await webappTabReplaysRememberedNavigation()).toBe(true)
    expect(request).not.toHaveBeenCalled()
  })

  it('the first browser tab takes the lock and restores', async () => {
    browserHostMock.mockReturnValue(true)
    installFakeLocks()

    // Browser host: the answer is not known until the lock is taken.
    expect(browserTabNavigationRole()).toBeNull()
    expect(await webappTabReplaysRememberedNavigation()).toBe(true)
  })

  it('a tab opened while another holds the lock boots like a peer', async () => {
    browserHostMock.mockReturnValue(true)
    installFakeLocks(new Set(['hermes.webapp.primary-tab']))

    expect(await webappTabReplaysRememberedNavigation()).toBe(false)
  })

  it('a browser without Web Locks keeps the restore rather than losing it', async () => {
    browserHostMock.mockReturnValue(true)

    expect(await webappTabReplaysRememberedNavigation()).toBe(true)
  })

  it('elects once per tab, not once per call', async () => {
    browserHostMock.mockReturnValue(true)
    const { request } = installFakeLocks()

    await webappTabReplaysRememberedNavigation()
    await webappTabReplaysRememberedNavigation()

    expect(request).toHaveBeenCalledTimes(1)
  })
})
