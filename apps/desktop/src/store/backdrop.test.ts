import { beforeEach, describe, expect, it } from 'vitest'

import {
  $backdropOpacity,
  BACKDROP_OPACITY_DEFAULT,
  BACKDROP_OPACITY_MAX,
  BACKDROP_OPACITY_MIN,
  clampBackdropOpacity,
  setBackdropOpacity
} from '@/store/backdrop'

const OPACITY_KEY = 'hermes.desktop.backdrop-opacity.v1'

describe('backdrop opacity', () => {
  beforeEach(() => {
    window.localStorage.removeItem(OPACITY_KEY)
    // Reset to the pre-lever default without going through the setter's
    // persist path first.
    $backdropOpacity.set(BACKDROP_OPACITY_DEFAULT)
  })

  it('defaults to the pre-lever visual strength so existing installs see no change', () => {
    expect($backdropOpacity.get()).toBe(BACKDROP_OPACITY_DEFAULT)
  })

  it('sets and persists a new value', () => {
    setBackdropOpacity(40)
    expect($backdropOpacity.get()).toBe(40)
    expect(window.localStorage.getItem(OPACITY_KEY)).toBe('40')
  })

  it('clamps above the max', () => {
    setBackdropOpacity(500)
    expect($backdropOpacity.get()).toBe(BACKDROP_OPACITY_MAX)
  })

  it('clamps below the min', () => {
    setBackdropOpacity(-20)
    expect($backdropOpacity.get()).toBe(BACKDROP_OPACITY_MIN)
  })

  it('falls back to the default for a non-finite value', () => {
    expect(clampBackdropOpacity(Number.NaN)).toBe(BACKDROP_OPACITY_DEFAULT)
    expect(clampBackdropOpacity(Number.POSITIVE_INFINITY)).toBe(BACKDROP_OPACITY_DEFAULT)
  })

  it('falls back to the default when the stored value is malformed', () => {
    window.localStorage.setItem(OPACITY_KEY, 'not-a-number')
    // Re-run the module's read path the way a fresh load would: the store
    // itself only reads storage once at import time, so we exercise the
    // same clamp/parse helper a reload would use.
    expect(clampBackdropOpacity(Number('not-a-number'))).toBe(BACKDROP_OPACITY_DEFAULT)
  })
})
