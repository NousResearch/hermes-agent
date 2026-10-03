import { describe, expect, it } from 'vitest'

import { luminance, normalizeHex } from './color'
import { skinToDesktopTheme } from './skin'

const withColors = (name: string, colors: Record<string, string>) => skinToDesktopTheme({ name, colors })

describe('skinToDesktopTheme', () => {
  it('returns null without a name or colors', () => {
    expect(skinToDesktopTheme({ name: 'x' })).toBeNull()
    expect(skinToDesktopTheme({ name: '', colors: { background: '#101010' } })).toBeNull()
  })

  it('maps the accent onto every brand token and keeps a single palette', () => {
    const theme = withColors('neon', { background: '#101020', ui_accent: '#ff33aa', banner_text: '#eeeeee' })!

    expect(theme.name).toBe('neon')
    expect(theme.colors.ring).toBe(theme.colors.primary)
    expect(theme.colors.midground).toBe(theme.colors.primary)
    // A skin is single-mode: the light/dark toggle must not invert it.
    expect(theme.colors).toBe(theme.darkColors)
  })

  it('seeds the background from status_bar_bg when none is explicit', () => {
    const theme = withColors('s', { status_bar_bg: '#0b0b0b', banner_text: '#ffffff' })!

    expect(theme.colors.background).toBe('#0b0b0b')
    expect(theme.colors.foreground).toBe('#ffffff')
  })

  it('buckets dark vs light from background luminance', () => {
    const dark = withColors('d', { background: '#111111', banner_text: '#eeeeee' })!
    const light = withColors('l', { background: '#fafafa', banner_text: '#111111' })!

    expect(luminance(dark.colors.background)).toBeLessThan(0.4)
    expect(luminance(light.colors.background)).toBeGreaterThan(0.4)
  })

  it('derives a dark base from light text when no background is given', () => {
    const theme = withColors('x', { banner_text: '#eeeeee', ui_accent: '#33ccff' })!

    expect(luminance(theme.colors.background)).toBeLessThan(0.4)
  })

  it('maps ui_error to destructive', () => {
    const theme = withColors('e', { background: '#101010', ui_error: '#ff5566' })!

    expect(theme.colors.destructive).toBe(normalizeHex('#ff5566'))
  })
})
