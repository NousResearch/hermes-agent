import { setAccentOverride } from './accent-override'
import { normalizeHex } from './color'
import { nousTheme } from './presets'
import { requestTheme } from './request'
import { retintTheme } from './retint'
import { installUserTheme } from './user-themes'

/**
 * Save an accent as the live profile's theme: a Nous copy retinted to `accent`, installed as a user theme.
 * `null` or Nous blue returns to the built-in Nous theme. Clears the unsaved `$accentOverride` preview.
 * An unparseable colour changes nothing.
 */
export function persistAccent(accent: null | string): void {
  const hex = accent === null ? null : normalizeHex(accent)

  if (accent !== null && !hex) {
    return
  }

  const theme =
    hex === null || hex === normalizeHex(nousTheme.colors.primary)
      ? nousTheme
      : installUserTheme({
          ...retintTheme(nousTheme, hex),
          name: `nous-accent-${hex.slice(1)}`,
          label: `${nousTheme.label} · ${hex}`
        })

  requestTheme(theme.name)
  setAccentOverride(null)
}
