import { beforeEach, describe, expect, it } from 'vitest'

import {
  $decorativeBackdrop,
  BACKDROP_OPACITY,
  type DecorativeBackdropMode,
  setDecorativeBackdrop
} from './backdrop'

const KEY = 'hermes.desktop.decorative-backdrop.v1'

describe('decorative backdrop preference', () => {
  beforeEach(() => {
    window.localStorage.clear()
    $decorativeBackdrop.set('full')
  })

  it.each([
    ['off', 0],
    ['subtle', 0.015],
    ['full', 0.025]
  ] as Array<[DecorativeBackdropMode, number]>)('maps %s to opacity %s', (mode, opacity) => {
    expect(BACKDROP_OPACITY[mode]).toBe(opacity)
  })

  it('defaults to full so the existing visual identity is unchanged', async () => {
    await import('./backdrop')

    expect($decorativeBackdrop.get()).toBe('full')
  })

  it('persists valid modes and ignores invalid modes', () => {
    setDecorativeBackdrop('subtle')
    expect($decorativeBackdrop.get()).toBe('subtle')
    expect(window.localStorage.getItem(KEY)).toBe('subtle')

    setDecorativeBackdrop('loud' as DecorativeBackdropMode)
    expect($decorativeBackdrop.get()).toBe('full')
  })
})
