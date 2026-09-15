import { beforeEach, describe, expect, it, vi } from 'vitest'

const GESTURES_KEY = 'hermes.desktop.composerPopout.gesturesEnabled'
const LEGACY_ENABLED_KEY = 'hermes.desktop.composerPopout.enabled'
const ZONES_KEY = 'hermes.desktop.composerPopout.zones.v1'

const loadStore = () => import('./composer-popout')

describe('composer pop-out preference', () => {
  beforeEach(() => {
    window.localStorage.clear()
    vi.resetModules()
  })

  it('docks every floating zone, preserves positions, and persists the lock', async () => {
    const first = await loadStore()

    first.setComposerPopoutPosition('left', { bottom: 100, right: 100 })
    first.setComposerPoppedOut('left', true)
    first.setComposerPopoutPosition('right', { bottom: 200, right: 200 })
    first.setComposerPoppedOut('right', true)

    first.setComposerPopoutGesturesEnabled(false)

    expect(first.$composerPopoutGesturesEnabled.get()).toBe(false)
    expect(first.getComposerPopoutZone('left')).toEqual({
      poppedOut: false,
      position: { bottom: 100, right: 100 }
    })
    expect(first.getComposerPopoutZone('right')).toEqual({
      poppedOut: false,
      position: { bottom: 200, right: 200 }
    })
    expect(window.localStorage.getItem(GESTURES_KEY)).toBe('false')
    expect(window.localStorage.getItem(LEGACY_ENABLED_KEY)).toBe('false')

    vi.resetModules()
    const reloaded = await loadStore()

    expect(reloaded.$composerPopoutGesturesEnabled.get()).toBe(false)
    expect(reloaded.getComposerPopoutZone('left').poppedOut).toBe(false)
    expect(reloaded.getComposerPopoutZone('right').poppedOut).toBe(false)
  })

  it('normalizes stale floating zones when the persisted preference is disabled', async () => {
    window.localStorage.setItem(GESTURES_KEY, 'false')
    window.localStorage.setItem(
      ZONES_KEY,
      JSON.stringify({ stale: { poppedOut: true, position: { bottom: 48, right: 64 } } })
    )

    const store = await loadStore()

    expect(store.getComposerPopoutZone('stale')).toEqual({
      poppedOut: false,
      position: { bottom: 48, right: 64 }
    })
  })

  it('keeps pop-out gestures enabled by default', async () => {
    const store = await loadStore()

    expect(store.$composerPopoutGesturesEnabled.get()).toBe(true)
  })
})
