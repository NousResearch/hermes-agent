import { describe, expect, it } from 'vitest'

import { contrastRatio } from './color'
import type { DesktopThemeColors } from './types'
import { adaptThemeColorsToWallpaper } from './wallpaper'

const base: DesktopThemeColors = {
  accent: '#f1f1f5',
  accentForeground: '#202024',
  background: '#ffffff',
  border: '#dedee5',
  card: '#ffffff',
  cardForeground: '#161616',
  destructive: '#b94a3a',
  destructiveForeground: '#ffffff',
  foreground: '#161616',
  input: '#e4e4e9',
  muted: '#f4f4f6',
  mutedForeground: '#68686e',
  popover: '#ffffff',
  popoverForeground: '#161616',
  primary: '#0053fd',
  primaryForeground: '#ffffff',
  ring: '#0053fd',
  secondary: '#ececf2',
  secondaryForeground: '#242428',
  sidebarBackground: '#fafafa',
  sidebarBorder: '#dedee5',
  userBubble: '#eeeeff',
  userBubbleBorder: '#d8d8e4'
}

describe('wallpaper-adaptive theme colors', () => {
  it('keeps brightness and text ownership while tinting accents and surfaces', () => {
    const adapted = adaptThemeColorsToWallpaper(base, { accent: '#e63658', dominant: '#84888e' })

    expect(adapted.background).toBe(base.background)
    expect(adapted.foreground).toBe(base.foreground)
    expect(adapted.destructive).toBe(base.destructive)
    expect(adapted.primary).not.toBe(base.primary)
    expect(adapted.sidebarBackground).not.toBe(base.sidebarBackground)
    expect(adapted.userBubble).not.toBe(base.userBubble)
    expect(contrastRatio(adapted.primary, adapted.sidebarBackground!)).toBeGreaterThanOrEqual(4.5)
    expect(contrastRatio(adapted.primary, adapted.primaryForeground)).toBeGreaterThanOrEqual(4.5)
  })

  it('retains the selected theme hue when the wallpaper is monochrome', () => {
    const adapted = adaptThemeColorsToWallpaper(base, { accent: '#777777', dominant: '#777777' })

    expect(adapted.primary).not.toBe('#777777')
    expect(contrastRatio(adapted.primary, adapted.sidebarBackground!)).toBeGreaterThanOrEqual(4.5)
  })

  it('applies a manually selected accent exactly while retaining readable button text', () => {
    const adapted = adaptThemeColorsToWallpaper(base, { accent: '#d45b9e', dominant: '#72808f' }, { exactAccent: true })

    expect(adapted.primary).toBe('#d45b9e')
    expect(adapted.ring).toBe('#d45b9e')
    expect(contrastRatio(adapted.primary, adapted.primaryForeground)).toBeGreaterThanOrEqual(4.5)
    expect(adapted.sidebarBackground).not.toBe('#72808f')
  })
})
