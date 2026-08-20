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

  it('keeps single-mode behavior when no paired palettes exist', () => {
    const theme = withColors('solo', { background: '#101020', ui_accent: '#ff33aa' })!

    // One shared object in both slots — the light/dark toggle must not invert it.
    expect(theme.colors).toBe(theme.darkColors)
  })

  it('ships distinct light/dark palettes when the skin has paired color blocks', () => {
    const theme = skinToDesktopTheme({
      name: 'dual',
      colors: { background: '#101020', ui_accent: '#ff33aa', banner_text: '#eeeeee' },
      light_colors: { background: '#fafafa', banner_text: '#111111', ui_accent: '#aa22cc' }
    })!

    expect(theme.colors).not.toBe(theme.darkColors)
    // Light slot uses the light_colors background; dark slot keeps the base.
    expect(luminance(theme.colors.background)).toBeGreaterThan(0.4)
    expect(luminance(theme.darkColors!.background)).toBeLessThan(0.4)
    // Both slots keep the authored accent identity, adapted per polarity.
    expect(theme.colors.primary).not.toBe(theme.darkColors!.primary)
  })

  it('treats a dark-authored base with only light_colors as dual-mode', () => {
    const theme = skinToDesktopTheme({
      name: 'darkfirst',
      colors: { background: '#0b0b0b', ui_accent: '#ffcc00', banner_text: '#ffffff' },
      light_colors: { background: '#ffffff', banner_text: '#222222', ui_accent: '#b8860b' }
    })!

    expect(luminance(theme.colors.background)).toBeGreaterThan(0.4)
    expect(luminance(theme.darkColors!.background)).toBeLessThan(0.4)
  })

  it('merges a fills-only paired block over the base palette', () => {
    const theme = skinToDesktopTheme({
      name: 'fills',
      colors: { background: '#0b0b0b', ui_accent: '#ffcc00', banner_text: '#ffffff' },
      // fills-only light overlay: flips the menu/status fills, keeps the vivid golds
      light_colors: { completion_menu_bg: '#ffffff', status_bar_bg: '#ffffff' }
    })!

    // Distinct palettes now, with the overlay's surface in the light slot.
    expect(theme.colors).not.toBe(theme.darkColors)
    expect(theme.colors.input).toBe(normalizeHex('#ffffff'))
  })

  it('buckets a chrome-only base by foreground luminance for polarity', () => {
    // No background/status_bar_bg: light text must read as a dark-authored base,
    // so the light_colors overlay lands in the light slot — the toggle works
    // instead of rendering the same palette in both modes.
    const theme = skinToDesktopTheme({
      name: 'chrome',
      colors: { ui_accent: '#ffcc00', banner_text: '#ffffff' },
      light_colors: { background: '#fafafa', banner_text: '#111111', ui_accent: '#b8860b' }
    })!

    expect(theme.colors).not.toBe(theme.darkColors)
    // Dark slot derives from the light-text chrome base (same bucket as buildPalette)…
    expect(luminance(theme.darkColors!.background)).toBeLessThan(0.4)
    // …and the light slot carries the overlay's canvas.
    expect(luminance(theme.colors.background)).toBeGreaterThan(0.4)
  })
})
