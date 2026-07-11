import { describe, expect, it } from 'vitest'

import {
  clampMenuBarTransparency,
  DEFAULT_MENU_BAR_TRANSPARENCY,
  getMenuBarSurfaceAlphas,
  menuBarTransparencyFromStorage
} from './menu-bar-transparency'

describe('menu bar transparency', () => {
  it('clamps and rounds values to the supported range', () => {
    expect(clampMenuBarTransparency(-12)).toBe(0)
    expect(clampMenuBarTransparency(44.6)).toBe(45)
    expect(clampMenuBarTransparency(140)).toBe(100)
  })

  it('preserves the approved glass treatment at the default value', () => {
    expect(DEFAULT_MENU_BAR_TRANSPARENCY).toBe(20)
    expect(getMenuBarSurfaceAlphas(DEFAULT_MENU_BAR_TRANSPARENCY)).toEqual({
      background: 0.86,
      gradientBottom: 0.72,
      gradientTop: 0.55,
      panel: 0.86
    })
  })

  it('uses the approved default only when no valid preference exists', () => {
    expect(menuBarTransparencyFromStorage(null)).toBe(DEFAULT_MENU_BAR_TRANSPARENCY)
    expect(menuBarTransparencyFromStorage('')).toBe(DEFAULT_MENU_BAR_TRANSPARENCY)
    expect(menuBarTransparencyFromStorage('not-a-number')).toBe(DEFAULT_MENU_BAR_TRANSPARENCY)
    expect(menuBarTransparencyFromStorage('0')).toBe(0)
    expect(menuBarTransparencyFromStorage('45')).toBe(45)
  })

  it('makes surfaces more transparent as the value increases', () => {
    const opaque = getMenuBarSurfaceAlphas(0)
    const transparent = getMenuBarSurfaceAlphas(100)

    expect(transparent.background).toBeLessThan(opaque.background)
    expect(transparent.gradientBottom).toBeLessThan(opaque.gradientBottom)
    expect(transparent.gradientTop).toBeLessThan(opaque.gradientTop)
    expect(transparent.panel).toBeLessThan(opaque.panel)
  })
})
