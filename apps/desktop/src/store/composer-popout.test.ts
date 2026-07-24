import { beforeEach, describe, expect, it, vi } from 'vitest'

const loadStore = () => import('./composer-popout')

describe('composer pop-out preference', () => {
  beforeEach(() => {
    window.localStorage.clear()
    vi.resetModules()
  })

  it('locks an already-floating composer to the dock and persists the preference', async () => {
    const first = await loadStore()
    first.setComposerPoppedOut(true)

    first.setComposerPopoutGesturesEnabled(false)

    expect(first.$composerPopoutGesturesEnabled.get()).toBe(false)
    expect(first.$composerPoppedOut.get()).toBe(false)
    expect(window.localStorage.getItem('hermes.desktop.composerPopout.gesturesEnabled')).toBe('false')
    expect(window.localStorage.getItem('hermes.desktop.composerPopout.enabled')).toBe('false')

    vi.resetModules()
    const reloaded = await loadStore()
    expect(reloaded.$composerPopoutGesturesEnabled.get()).toBe(false)
    expect(reloaded.$composerPoppedOut.get()).toBe(false)
  })

  it('keeps pop-out gestures enabled by default', async () => {
    const store = await loadStore()

    expect(store.$composerPopoutGesturesEnabled.get()).toBe(true)
  })
})
