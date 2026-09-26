import { beforeEach, describe, expect, it, vi } from 'vitest'

const GESTURES_KEY = 'hermes.desktop.composerPopout.gesturesEnabled'
const ZONES_KEY = 'hermes.desktop.composerPopout.zones.v1'

const loadStore = () => import('./composer-popout')

describe('composer pop-out preference', () => {
  beforeEach(() => {
    window.localStorage.clear()
    vi.resetModules()
  })

  it('docks the shared composer, preserves its placement, and persists the lock', async () => {
    const first = await loadStore()

    first.setComposerPopoutPosition({ bottom: 200, right: 200 })
    first.setComposerPoppedOut(true)

    first.setComposerPopoutGesturesEnabled(false)

    expect(first.$composerPopoutGesturesEnabled.get()).toBe(false)
    expect(first.$composerPopout.get()).toEqual({
      poppedOut: false,
      position: { bottom: 200, right: 200 }
    })
    expect(window.localStorage.getItem(GESTURES_KEY)).toBe('false')

    vi.resetModules()
    const reloaded = await loadStore()

    expect(reloaded.$composerPopoutGesturesEnabled.get()).toBe(false)
    expect(reloaded.$composerPopout.get()).toEqual(first.$composerPopout.get())
  })

  it('normalizes stale floating zones when the persisted preference is disabled', async () => {
    window.localStorage.setItem(GESTURES_KEY, 'false')
    window.localStorage.setItem(
      ZONES_KEY,
      JSON.stringify({ stale: { poppedOut: true, position: { bottom: 48, right: 64 } } })
    )

    const store = await loadStore()

    expect(store.$composerPopout.get()).toEqual({
      poppedOut: false,
      position: { bottom: 48, right: 64 }
    })
  })
})
