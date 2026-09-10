/**
 * Hermes skin → DesktopTheme converter.
 *
 * A "skin" is the CLI/TUI theme unit: a YAML file in `$HERMES_HOME/skins/` (or a
 * built-in) resolved by `hermes_cli/skin_engine.py` and pushed to every surface
 * over JSON-RPC (`gateway.ready`, `skin.changed`, `config.get skin`). This is the
 * one place the desktop turns that CLI-shaped palette into a `DesktopTheme`, so a
 * skin Hermes authors from a prompt lights up all three surfaces from one file.
 *
 * Skins carry terminal-oriented keys (banner/status/completion). We seed the
 * desktop model from the load-bearing few (background, foreground, accent, error)
 * and derive every glass/shadcn surface by mixing toward bg/fg — the same "naive
 * token converter" strategy as the VS Code importer. A skin is single-mode, so
 * both `colors` and `darkColors` get the converted palette; `renderedModeFor`
 * still picks `.dark` from the real background luminance.
 */

import type { HermesSkin, SkinColors } from '@hermes/shared/skin'

import { ensureContrast, luminance, mix, normalizeHex, readableOn } from './color'
import type { DesktopTheme, DesktopThemeColors } from './types'

// The accent labels the sidebar in small uppercase text, so it must clear WCAG AA
// for normal text or section headers go invisible — mirrors the VS Code importer.
const ACCENT_MIN_CONTRAST = 4.5

/** First normalizable hex among `keys`, alpha flattened over `backdrop`. */
const pick = (colors: SkinColors, keys: string[], backdrop: string): string | null => {
  for (const key of keys) {
    const value = normalizeHex(colors[key], backdrop)

    if (value) {
      return value
    }
  }

  return null
}

const titleCase = (name: string): string => name.charAt(0).toUpperCase() + name.slice(1)

/**
 * Build one `DesktopThemeColors` palette from a skin color block.
 *
 * Extracted from the old single-palette body so a dual-mode skin can derive
 * its light and dark palettes independently (each with its own polarity).
 */
function buildPalette(colors: SkinColors): DesktopThemeColors {
  // Background is the backdrop every other token flattens alpha over. Skins are
  // terminal-first so most only tint chrome — `status_bar_bg` is the closest
  // thing to an app surface; `background` is the explicit opt-in for GUI authors.
  const seededBg = pick(colors, ['background', 'status_bar_bg'], '#000000')
  const foregroundSeed = pick(colors, ['ui_text', 'banner_text', 'status_bar_text'], seededBg ?? '#000000')

  // No background given: bucket by foreground luminance (light text ⇒ dark app).
  const background = seededBg ?? (foregroundSeed && luminance(foregroundSeed) > 0.5 ? '#141414' : '#f7f7f8')
  const dark = luminance(background) < 0.4
  const foreground = foregroundSeed ?? (dark ? '#e6e6e6' : '#161616')

  const accentSeed =
    pick(colors, ['ui_accent', 'banner_accent', 'banner_title'], background) ?? mix(foreground, background, 0.55)

  const sidebar = mix(background, foreground, dark ? 0.02 : 0.012)
  const accent = ensureContrast(accentSeed, sidebar, ACCENT_MIN_CONTRAST)

  const border =
    pick(colors, ['ui_border', 'banner_border'], background) ?? mix(background, foreground, dark ? 0.16 : 0.14)

  const mutedForeground =
    pick(colors, ['banner_dim', 'session_border'], background) ?? mix(foreground, background, 0.45)

  const destructive = pick(colors, ['ui_error'], background) ?? '#e25563'

  return {
    background,
    foreground,
    card: mix(background, foreground, dark ? 0.04 : 0.025),
    cardForeground: foreground,
    muted: mix(background, foreground, dark ? 0.06 : 0.04),
    mutedForeground,
    popover: mix(background, foreground, dark ? 0.08 : 0.05),
    popoverForeground: foreground,
    primary: accent,
    primaryForeground: readableOn(accent),
    secondary: mix(accent, background, dark ? 0.72 : 0.86),
    secondaryForeground: foreground,
    accent: mix(accent, background, dark ? 0.82 : 0.88),
    accentForeground: foreground,
    border,
    input: pick(colors, ['completion_menu_bg'], background) ?? mix(background, foreground, dark ? 0.1 : 0.06),
    ring: accent,
    midground: accent,
    midgroundForeground: readableOn(accent),
    composerRing: accent,
    destructive,
    destructiveForeground: readableOn(destructive),
    sidebarBackground: sidebar,
    sidebarBorder: border,
    userBubble: mix(background, accent, dark ? 0.18 : 0.12),
    userBubbleBorder: border
  }
}

/** True when a paired palette block carries at least one real token. */
const hasTokens = (block: SkinColors | undefined): boolean => !!block && Object.keys(block).length > 0

/**
 * Convert a resolved skin into a `DesktopTheme`, or null when it carries no
 * usable colors (so a broken/empty skin never registers junk).
 *
 * A skin may ship a base `colors` block plus hand-tuned `light_colors` /
 * `dark_colors` overlays (the TUI's paired-palette contract). When paired
 * blocks exist, the base palette's polarity decides which overlay belongs to
 * which mode, and the desktop produces genuinely distinct `colors` (light)
 * and `darkColors` (dark) — so the light/dark toggle works. Without paired
 * blocks the skin is single-mode: both slots get the same palette, exactly
 * as before.
 */
export function skinToDesktopTheme(skin: HermesSkin): DesktopTheme | null {
  const name = (skin.name ?? '').trim()
  const colors = skin.colors

  if (!name || !colors || typeof colors !== 'object') {
    return null
  }

  const hasLight = hasTokens(skin.light_colors)
  const hasDark = hasTokens(skin.dark_colors)

  // Single-mode skin (no paired palettes): identical palette in both slots —
  // the light/dark toggle must not invert it. Kept as one shared object so
  // callers comparing by reference see the same palette as before.
  if (!hasLight && !hasDark) {
    const palette = buildPalette(colors)

    return {
      name,
      label: titleCase(name),
      description: 'Hermes skin',
      colors: palette,
      darkColors: palette
    }
  }

  // Dual-mode: the base palette's authored background decides polarity (mirror
  // of the TUI's `skinIsLight`). The opposite mode is the base merged with the
  // matching hand-tuned overlay; the authored mode is the base itself.
  //
  // A chrome-only base (no `background`/`status_bar_bg`) must fall back to the
  // SAME foreground-luminance bucket `buildPalette` uses, or the polarity seed
  // disagrees with the palette it derives: light text ⇒ dark canvas ⇒ the
  // light_colors overlay belongs in the light slot.
  const seededBg = pick(colors, ['background', 'status_bar_bg'], '#000000')
  const foregroundSeed = pick(colors, ['ui_text', 'banner_text', 'status_bar_text'], seededBg ?? '#000000')
  const baseDark = seededBg
    ? luminance(seededBg) < 0.4
    : (foregroundSeed ? luminance(foregroundSeed) > 0.5 : false)

  const lightSource = baseDark ? { ...colors, ...(skin.light_colors ?? {}) } : colors
  const darkSource = baseDark ? colors : { ...colors, ...(skin.dark_colors ?? {}) }

  return {
    name,
    label: titleCase(name),
    description: 'Hermes skin',
    colors: buildPalette(lightSource),
    darkColors: buildPalette(darkSource)
  }
}
