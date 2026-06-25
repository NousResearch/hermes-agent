import { beforeEach, describe, expect, it, vi } from 'vitest'

const LEGACY_ENABLED_KEY = 'hermes.desktop.composerPopout.enabled'
const LEGACY_POSITION_KEY = 'hermes.desktop.composerPopout.position'
const profileEnabledKey = (profile: string) => `hermes.desktop.composerPopout.v2.${encodeURIComponent(profile)}.enabled`
const profilePositionKey = (profile: string) => `hermes.desktop.composerPopout.v2.${encodeURIComponent(profile)}.position`

async function loadStores() {
  const profile = await import('./profile')
  const popout = await import('./composer-popout')

  return { popout, profile }
}

describe('composer popout profile-scoped persistence', () => {
  beforeEach(() => {
    vi.resetModules()
    window.localStorage.clear()
    Object.defineProperty(window, 'innerWidth', { configurable: true, value: 1440 })
    Object.defineProperty(window, 'innerHeight', { configurable: true, value: 900 })
  })

  it('does not restore the legacy global popped-out flag for every profile', async () => {
    window.localStorage.setItem(LEGACY_ENABLED_KEY, 'true')
    window.localStorage.setItem(LEGACY_POSITION_KEY, JSON.stringify({ bottom: 42, right: 1297 }))

    const { popout } = await loadStores()

    expect(popout.$composerPoppedOut.get()).toBe(false)
    expect(popout.$composerPopoutPosition.get()).toEqual({ bottom: 42, right: 1120 })
  })

  it('persists popped-out state and position under the active gateway profile only', async () => {
    const { popout, profile } = await loadStores()

    profile.$activeGatewayProfile.set('acewill-dev')
    popout.setComposerPoppedOut(true)
    popout.setComposerPopoutPosition({ bottom: 64, right: 72 }, { persist: true })

    expect(window.localStorage.getItem(profileEnabledKey('acewill-dev'))).toBe('true')
    expect(window.localStorage.getItem(profilePositionKey('acewill-dev'))).toBe(JSON.stringify({ bottom: 64, right: 72 }))
    expect(window.localStorage.getItem(LEGACY_ENABLED_KEY)).toBeNull()

    profile.$activeGatewayProfile.set('aivideo-dev')
    expect(popout.$composerPoppedOut.get()).toBe(false)
    expect(popout.$composerPopoutPosition.get()).toEqual({ bottom: 24, right: 24 })

    profile.$activeGatewayProfile.set('acewill-dev')
    expect(popout.$composerPoppedOut.get()).toBe(true)
    expect(popout.$composerPopoutPosition.get()).toEqual({ bottom: 64, right: 72 })
  })
})
