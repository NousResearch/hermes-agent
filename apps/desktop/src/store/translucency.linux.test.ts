// @vitest-environment jsdom
import { describe, expect, it, vi } from 'vitest'

// Linux, as the preload bridge reports it: neither glass nor translucency.
// Pinned before the store module evaluates (hoisted above the imports) so
// TRANSLUCENCY_SUPPORTED resolves false, and the bridge records every state
// the renderer hands main.
const sent = vi.hoisted(() => {
  const calls: unknown[] = []

  Object.defineProperty(globalThis.navigator, 'platform', { configurable: true, value: 'Linux x86_64' })
  Object.defineProperty(globalThis.window, 'hermesDesktop', {
    configurable: true,
    value: {
      glassSupported: false,
      translucencySupported: false,
      setTranslucency: (state: unknown) => calls.push(state)
    }
  })

  return calls
})

import { type TranslucencyState, windowOpacityFor } from '@hermes/shared/translucency'

import { $translucency, $translucencyBook, setAppearance, setTranslucency, TRANSLUCENCY_SUPPORTED } from './translucency'

// Settings hides the row on Linux, so nothing the store resolves may fade the
// window: not the 29% platform default an untouched profile lands on, and not
// a value that arrives from anywhere else.
describe('translucency where Settings hides the row', () => {
  it('resolves to an opaque window and tells main exactly that', () => {
    expect(TRANSLUCENCY_SUPPORTED).toBe(false)

    setTranslucency(60)

    for (const state of [$translucency.get(), ...(sent as TranslucencyState[])]) {
      expect(windowOpacityFor(state)).toBe(1)
    }

    expect(sent.length).toBeGreaterThan(0)
    expect(document.documentElement.hasAttribute('data-hermes-clear')).toBe(false)
  })

  // The regression this whole fix answers: an untouched profile resolves the
  // 29% Clear platform default, and that default reached the window. The
  // empty-book ladder must land on OFF, in both appearances — not on the
  // default that only exists for platforms offered the row.
  it('resolves an untouched fresh profile to off, in both appearances', () => {
    // An empty book is the untouched shape (the win10 test resets the same way).
    $translucencyBook.set({ mode: 'clear', base: {}, light: {}, dark: {} })
    expect($translucencyBook.get()).toEqual({ mode: 'clear', base: {}, light: {}, dark: {} })

    for (const appearance of ['light', 'dark'] as const) {
      setAppearance(appearance)

      expect($translucency.get().intensity).toBe(0)
      expect($translucency.get().fade).toBe(0)
      expect(windowOpacityFor($translucency.get())).toBe(1)
    }
  })

  // Main persists the state it is handed verbatim, so translucency.json only
  // heals from a stale 29 if the EXACT value over the wire is off — opaque-by-
  // another-route would leave the file (and the first pre-render frame) faded.
  it('hands main the exact off state translucency.json heals to', () => {
    setAppearance('dark')
    setTranslucency(29)

    expect(sent.length).toBeGreaterThan(0)

    for (const state of sent as TranslucencyState[]) {
      expect(state.mode).toBe('clear')
      expect(state.intensity).toBe(0)
      expect(state.fade).toBe(0)
    }
  })

  // Whatever an edit or a migration puts in the book, the resolved answer on
  // this platform is the same: the whole ramp reads fully opaque.
  it('keeps the window opaque at every intensity, in both appearances', () => {
    for (const appearance of ['light', 'dark'] as const) {
      setAppearance(appearance)

      for (const intensity of [0, 1, 29, 50, 100]) {
        setTranslucency(intensity)

        expect($translucency.get().intensity).toBe(0)
        expect(windowOpacityFor($translucency.get())).toBe(1)
      }
    }
  })
})
