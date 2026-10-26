// @vitest-environment jsdom
import { describe, expect, it, vi } from 'vitest'

// Simulate Linux: pin the platform AND tell the preload bridge that neither half
// of the feature exists there (Electron answers both from the process platform).
// Both must be in place BEFORE the store module is evaluated (hoisted above the
// imports) so GLASS_SUPPORTED and TRANSLUCENCY_SUPPORTED resolve false.
//
// That is the exact shape of the bug this test guards: `translucencySupportedOn`
// reports false on Linux, so Settings hides the lever — but the store still
// resolved the macOS 29% default and mirrored it to main, main persisted it, and
// Electron's X11 path honours `setOpacity` (_NET_WM_WINDOW_OPACITY) even though
// the docs call it a no-op on Linux. Every fresh Linux window was therefore born
// ~94% opaque with no row anywhere to undo it.
vi.hoisted(() => {
  Object.defineProperty(globalThis.navigator, 'platform', { configurable: true, value: 'Linux x86_64' })
  Object.defineProperty(globalThis.window, 'hermesDesktop', {
    configurable: true,
    value: { glassSupported: false, translucencySupported: false, setTranslucency: vi.fn() }
  })
})

import {
  $translucency,
  GLASS_SUPPORTED,
  setTranslucency,
  setTranslucencyMode,
  TRANSLUCENCY_SUPPORTED
} from './translucency'

const bridge = window.hermesDesktop as unknown as { setTranslucency: ReturnType<typeof vi.fn> }

describe('Linux translucency (no lever, so no fade)', () => {
  it('runs its assertions for real in this environment', () => {
    // Guard the guard: if the platform pin or the preload answer ever stops
    // landing, the suite below silently tests a Mac instead of a Linux box.
    expect(navigator.platform).toBe('Linux x86_64')
    expect(GLASS_SUPPORTED).toBe(false)
    expect(TRANSLUCENCY_SUPPORTED).toBe(false)
  })

  // The leak's fuel: nothing on Linux tunes this, yet the resolved state still
  // carries the macOS default that main turns into a window opacity.
  it('still resolves the macOS default tint — which is exactly why the bridge is gated', () => {
    expect($translucency.get().intensity).toBe(29)
    expect($translucency.get().mode).toBe('clear')
  })

  it('never mirrors the resolved state to main', () => {
    // The subscription fires once at import, before this test could clear the
    // spy — so a non-zero count here is the import-time send itself.
    expect(bridge.setTranslucency).not.toHaveBeenCalled()

    setTranslucency(40)
    setTranslucencyMode('glass')
    setTranslucency(0)

    expect(bridge.setTranslucency).not.toHaveBeenCalled()
  })
})
