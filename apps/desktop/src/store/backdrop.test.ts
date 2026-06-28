import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { DecorativeBackdropMode } from './backdrop'

const KEY = 'hermes.desktop.decorative-backdrop.v1'

async function loadBackdropStore(stored?: string) {
  window.localStorage.clear()

  if (stored !== undefined) {
    window.localStorage.setItem(KEY, stored)
  }

  vi.resetModules()

  return import('./backdrop')
}

describe('decorative backdrop preference', () => {
  beforeEach(() => {
    window.localStorage.clear()
    vi.resetModules()
  })

  it.each([
    ['off', 0],
    ['subtle', 0.015],
    ['full', 0.025]
  ] as Array<[DecorativeBackdropMode, number]>)('maps %s to opacity %s', async (mode, opacity) => {
    const { BACKDROP_OPACITY } = await loadBackdropStore()

    expect(BACKDROP_OPACITY[mode]).toBe(opacity)
  })

  it('defaults to full so the existing visual identity is unchanged', async () => {
    const { $decorativeBackdrop } = await loadBackdropStore()

    expect($decorativeBackdrop.get()).toBe('full')
  })

  it('initializes from a valid stored mode', async () => {
    const { $decorativeBackdrop } = await loadBackdropStore('subtle')

    expect($decorativeBackdrop.get()).toBe('subtle')
  })

  it('falls back to full for an invalid stored mode', async () => {
    const { $decorativeBackdrop } = await loadBackdropStore('loud')

    expect($decorativeBackdrop.get()).toBe('full')
  })

  it('persists valid modes and ignores invalid modes', async () => {
    const { $decorativeBackdrop, setDecorativeBackdrop } = await loadBackdropStore()

    setDecorativeBackdrop('subtle')
    expect($decorativeBackdrop.get()).toBe('subtle')
    expect(window.localStorage.getItem(KEY)).toBe('subtle')

    setDecorativeBackdrop('loud' as DecorativeBackdropMode)
    expect($decorativeBackdrop.get()).toBe('full')
    expect(window.localStorage.getItem(KEY)).toBe('full')
  })
})
