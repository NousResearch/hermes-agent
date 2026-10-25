// @vitest-environment jsdom
import { describe, expect, it, vi } from 'vitest'

// Simulate Linux: pin the platform AND tell the preload bridge that
// translucency is unsupported. Both must be in place BEFORE the store module
// is evaluated (hoisted above the imports), so TRANSLUCENCY_SUPPORTED
// resolves false — exactly as the preload reports it on Linux
// (main answers `translucencySupportedOn(process.platform)`, which is false
// for anything but darwin/win32).
//
// This is the shape of the bug this file guards. Electron 44.9.3 made
// `setOpacity` WORK on Linux — it now writes X11's `_NET_WM_WINDOW_OPACITY`
// — while the setting that drives it stayed hidden there. Main kept asking
// for a faded window, so the DEFAULT state (intensity 29, clear) rendered the
// chat window at ~94% opacity with the desktop wallpaper bleeding through it,
// and no UI could turn it off. The renderer half of that is the Clear-mode
// page effect: `data-hermes-clear` strengthens the overlay scrim to
// compensate for a window that is ACTUALLY faded — and on Linux it never is.
vi.hoisted(() => {
  Object.defineProperty(globalThis.navigator, 'platform', { configurable: true, value: 'Linux x86_64' })
  Object.defineProperty(globalThis.window, 'hermesDesktop', {
    configurable: true,
    value: { glassSupported: false, translucencySupported: false }
  })
})

import { $translucency, $translucencyBook, setAppearance, setTranslucency, TRANSLUCENCY_SUPPORTED } from './translucency'

const clearAttr = () => document.documentElement.hasAttribute('data-hermes-clear')
const glassAttr = () => document.documentElement.hasAttribute('data-hermes-glass')

describe('Linux translucency (regression: Electron 44.9.3 made setOpacity real)', () => {
  it('runs its assertions for real in this environment', () => {
    // Guard the guard: if the platform pin or the bridge ever stops landing,
    // every assertion below silently tests the wrong environment.
    expect(navigator.platform).toBe('Linux x86_64')
    expect(TRANSLUCENCY_SUPPORTED).toBe(false)
  })

  it('never marks the page as Clear — there is no fade to compensate for', () => {
    setAppearance('dark')
    setTranslucency(29)

    expect(clearAttr()).toBe(false)
  })

  it('never marks the page as Glass — there is no native material to read', () => {
    setAppearance('dark')
    setTranslucency(60)

    expect(glassAttr()).toBe(false)
  })

  it('leaves the page unmarked at every intensity, in both appearances', () => {
    for (const appearance of ['light', 'dark'] as const) {
      setAppearance(appearance)

      for (const intensity of [0, 1, 29, 50, 100]) {
        setTranslucency(intensity)

        expect(clearAttr(), `${appearance} ${intensity}`).toBe(false)
        expect(glassAttr(), `${appearance} ${intensity}`).toBe(false)
      }
    }
  })

  it('keeps the resolved state readable — the setting is hidden, not deleted', () => {
    // The state still resolves (main still persists it, and the UI still
    // reports it); only the NATIVE effect and the page effect are refused.
    // Hiding the row must not corrupt the value the app carries.
    setAppearance('dark')
    setTranslucency(29)

    expect($translucency.get().intensity).toBe(29)
    expect($translucencyBook.get().mode).toBe('clear')
  })
})
