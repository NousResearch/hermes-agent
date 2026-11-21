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

import { $translucency, setTranslucency, TRANSLUCENCY_SUPPORTED } from './translucency'

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
})
