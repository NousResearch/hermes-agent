// @vitest-environment jsdom
import { describe, expect, it, vi } from 'vitest'

// Simulate Linux: pin the platform AND tell the preload bridge that neither glass
// nor translucency exists there (`translucencySupportedOn` is darwin/win32 only).
// Both must be in place BEFORE the store module is evaluated (hoisted above the
// imports) so TRANSLUCENCY_SUPPORTED resolves false.
//
// That is the shape of the bug this test guards: `translucencySupportedOn` gated
// the Settings row and NOTHING else, so a Linux profile still inherited the mac
// glass default (29), resolved it as clear mode, and faded the whole window —
// text included — with no Settings row anywhere to undo it.
vi.hoisted(() => {
  Object.defineProperty(globalThis.navigator, 'platform', { configurable: true, value: 'Linux x86_64' })
  Object.defineProperty(globalThis.window, 'hermesDesktop', {
    configurable: true,
    value: { glassSupported: false, translucencySupported: false }
  })
})

import { $translucency, $translucencyBook, GLASS_SUPPORTED, setAppearance, setTranslucency, TRANSLUCENCY_SUPPORTED } from './translucency'

const clearAttr = () => document.documentElement.hasAttribute('data-hermes-clear')

describe('translucency on a platform that has none', () => {
  it('runs its assertions for real in this environment', () => {
    // Guard the guard: if the platform pin or the bridge stub ever stops landing,
    // every assertion below silently tests a supported platform instead.
    expect(navigator.platform).toBe('Linux x86_64')
    expect(TRANSLUCENCY_SUPPORTED).toBe(false)
    expect(GLASS_SUPPORTED).toBe(false)
  })

  it('resolves an untouched profile to off, not to the mac glass preset', () => {
    expect($translucency.get()).toMatchObject({ intensity: 0, fade: 0, mode: 'clear' })

    setAppearance('light')
    expect($translucency.get().intensity).toBe(0)
    setAppearance('dark')
  })

  it('resolves a persisted book with a non-zero intensity to off', () => {
    // The exact file the upgrade left behind: `{ intensity: 29, mode: 'clear' }`.
    $translucencyBook.set({ mode: 'clear', base: { intensity: 29 }, light: {}, dark: {} })

    expect($translucency.get().intensity).toBe(0)
  })

  it('paints no clear-mode overlay scrim, because there is no fade to compensate for', () => {
    // The store drives the attribute from the RESOLVED state, so raising the lever
    // by hand (which the UI does not offer here) still must not mark the window.
    setTranslucency(80)

    expect($translucency.get().intensity).toBe(0)
    expect(clearAttr()).toBe(false)
  })
})
