import { contrastRatio } from '@hermes/shared/color'
import { describe, expect, it } from 'vitest'

import { portalTheme } from './portal'

describe('Portal desktop legibility', () => {
  it.each([portalTheme.colors, portalTheme.darkColors!])('keeps text readable on its surfaces', colors => {
    const pairs = [
      [colors.foreground, colors.background],
      [colors.mutedForeground, colors.background],
      [colors.mutedForeground, colors.card],
      [colors.primaryForeground, colors.primary],
      [colors.popoverForeground, colors.popover],
      [colors.secondaryForeground, colors.secondary],
      [colors.destructiveForeground, colors.destructive]
    ]

    for (const [foreground, background] of pairs) {
      expect(contrastRatio(foreground, background)).toBeGreaterThanOrEqual(4.5)
    }
  })
})
