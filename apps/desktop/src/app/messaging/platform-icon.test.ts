import { describe, expect, it } from 'vitest'

import { ensureVisible, luminance } from './platform-icon'

describe('luminance', () => {
  it('returns 0 for pure black', () => {
    expect(luminance('#000000')).toBe(0)
  })

  it('returns 1 for pure white', () => {
    expect(luminance('#ffffff')).toBeCloseTo(1, 5)
  })

  it('returns mid-range for medium gray', () => {
    const lum = luminance('#808080')
    expect(lum).toBeGreaterThan(0.15)
    expect(lum).toBeLessThan(0.3)
  })

  it('handles invalid hex gracefully', () => {
    expect(luminance('not-a-hex')).toBe(0.5)
    expect(luminance('#zzzzzz')).toBe(0.5)
  })
})

describe('ensureVisible', () => {
  it('returns original colour in light mode', () => {
    expect(ensureVisible('#000000', false)).toBe('#000000')
    expect(ensureVisible('#4A154B', false)).toBe('#4A154B')
  })

  it('returns original colour when already bright enough in dark mode', () => {
    // #26A5E4 (Telegram blue) has high luminance
    expect(ensureVisible('#26A5E4', true)).toBe('#26A5E4')
    expect(ensureVisible('#5865F2', true)).toBe('#5865F2')
  })

  it('lightens Matrix black (#000000) in dark mode', () => {
    const result = ensureVisible('#000000', true)
    expect(result).not.toBe('#000000')
    // Result should be a medium gray
    expect(luminance(result)).toBeGreaterThanOrEqual(0.1)
    expect(luminance(result)).toBeLessThan(0.2)
  })

  it('lightens Slack dark purple (#4A154B) in dark mode', () => {
    const result = ensureVisible('#4A154B', true)
    expect(result).not.toBe('#4A154B')
    expect(luminance(result)).toBeGreaterThanOrEqual(0.1)
  })

  it('preserves hex format', () => {
    const result = ensureVisible('#000000', true)
    expect(result).toMatch(/^#[0-9a-f]{6}$/)
  })

  it('handles invalid hex gracefully', () => {
    expect(ensureVisible('invalid', true)).toBe('invalid')
    expect(ensureVisible('invalid', false)).toBe('invalid')
  })

  it('all PLATFORM_ICONS brand colours are visible in dark mode', () => {
    // Simulate the full set of brand colours used in the component.
    const brandColors = [
      '#26A5E4', // telegram
      '#5865F2', // discord
      '#4A154B', // slack
      '#0058CC', // mattermost
      '#000000', // matrix
      '#3A76F0', // signal
      '#25D366', // whatsapp
      '#0BD318', // bluebubbles
      '#18BCF2', // homeassistant
      '#EA4335', // email
      '#F43F5E', // sms
      '#71717A', // webhook
      '#64748B', // api_server
      '#07C160', // weixin
      '#EB1923', // qqbot
      '#FB7299' // yuanbao
    ]

    for (const color of brandColors) {
      const adjusted = ensureVisible(color, true)
      expect(luminance(adjusted)).toBeGreaterThanOrEqual(0.1)
    }
  })
})
