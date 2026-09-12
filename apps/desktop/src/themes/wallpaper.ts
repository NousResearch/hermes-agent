import type { WallpaperPalette } from '@/lib/wallpaper-palette'

import { contrastRatio, ensureContrast, hexToRgb, mix, relativeLuminance } from './color'
import type { DesktopThemeColors } from './types'

const ACCENT_TEXT_CONTRAST = 4.5

interface WallpaperThemeOptions {
  exactAccent?: boolean
}

function saturation(color: string): number {
  const rgb = hexToRgb(color)

  if (!rgb) {
    return 0
  }

  const max = Math.max(...rgb)
  const min = Math.min(...rgb)

  return max === 0 ? 0 : (max - min) / max
}

function readableForeground(background: string): string {
  const dark = '#161616'
  const light = '#ffffff'

  return contrastRatio(dark, background) >= contrastRatio(light, background) ? dark : light
}

/**
 * Tint a theme from wallpaper colors without changing its brightness contract.
 * Text and the main background remain owned by the selected skin; image colors
 * influence accents and low-amplitude surfaces only.
 */
export function adaptThemeColorsToWallpaper(
  base: DesktopThemeColors,
  palette: WallpaperPalette,
  options: WallpaperThemeOptions = {}
): DesktopThemeColors {
  const dark = relativeLuminance(base.background) < 0.4
  const surfaceMix = dark ? 0.16 : 0.1
  const softMix = dark ? 0.2 : 0.13
  const sidebarBackground = mix(base.sidebarBackground ?? base.background, palette.dominant, surfaceMix)

  const accentSeed = options.exactAccent
    ? palette.accent
    : saturation(palette.accent) >= 0.16
      ? mix(base.primary, palette.accent, 0.82)
      : mix(base.primary, palette.dominant, 0.3)

  // Automatic colors are contrast-normalized because their source is
  // uncontrolled. A manually selected accent remains exact; its foreground is
  // still chosen for readability below, so button labels remain legible.
  const primary = options.exactAccent ? accentSeed : ensureContrast(accentSeed, sidebarBackground, ACCENT_TEXT_CONTRAST)
  const secondary = mix(base.secondary, palette.dominant, softMix)
  const accent = mix(base.accent, palette.dominant, softMix)
  const card = mix(base.card, palette.dominant, dark ? 0.1 : 0.06)
  const popover = mix(base.popover, palette.dominant, dark ? 0.08 : 0.05)
  const userBubble = mix(base.userBubble ?? base.popover, palette.accent, dark ? 0.18 : 0.12)

  return {
    ...base,
    accent,
    accentForeground: ensureContrast(base.accentForeground, accent, ACCENT_TEXT_CONTRAST),
    card,
    cardForeground: ensureContrast(base.cardForeground, card, ACCENT_TEXT_CONTRAST),
    composerRing: primary,
    input: mix(base.input, palette.dominant, dark ? 0.12 : 0.08),
    midground: primary,
    midgroundForeground: readableForeground(primary),
    muted: mix(base.muted, palette.dominant, dark ? 0.1 : 0.06),
    popover,
    popoverForeground: ensureContrast(base.popoverForeground, popover, ACCENT_TEXT_CONTRAST),
    primary,
    primaryForeground: readableForeground(primary),
    ring: primary,
    secondary,
    secondaryForeground: ensureContrast(base.secondaryForeground, secondary, ACCENT_TEXT_CONTRAST),
    sidebarBackground,
    sidebarBorder: mix(base.sidebarBorder ?? base.border, primary, 0.16),
    userBubble,
    userBubbleBorder: mix(base.userBubbleBorder ?? base.border, primary, 0.2)
  }
}
