/**
 * Runtime validity bar for a `DesktopTheme` arriving from OUTSIDE the bundle —
 * a stored user install, a cached backend skin, a plugin contribution.
 *
 * Its own module so every consumer shares one bar without an import cycle:
 * `user-themes` imports `backend-sync` (for the merged registry), so the guard
 * cannot live in either of them.
 */

import type { DesktopTheme, DesktopThemeColors } from './types'

// Kept loose on purpose — `applyTheme` tolerates missing optionals via
// fallbacks — but a theme with no background/foreground/primary is junk.
const REQUIRED_COLOR_KEYS: ReadonlyArray<keyof DesktopThemeColors> = ['background', 'foreground', 'primary']

export function isValidTheme(value: unknown): value is DesktopTheme {
  if (!value || typeof value !== 'object') {
    return false
  }

  const theme = value as Partial<DesktopTheme>

  if (typeof theme.name !== 'string' || typeof theme.label !== 'string' || !theme.colors) {
    return false
  }

  const colors = theme.colors as unknown as Record<string, unknown>

  return REQUIRED_COLOR_KEYS.every(key => typeof colors[key] === 'string')
}
