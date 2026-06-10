/**
 * VS Code color-theme → DesktopTheme converter.
 *
 * VS Code themes carry ~hundreds of `workbench.colorCustomization` keys, but the
 * desktop theme model only needs a `DesktopThemeColors` struct — `applyTheme`
 * derives every glass/shadcn token from a small seed chain via `color-mix()`.
 * In practice ~6 workbench keys carry the whole look (background, foreground,
 * accent, elevated surface, sidebar, error); everything else we derive by mixing
 * those toward the background/foreground. That's the "naive token converter".
 *
 * A VS Code theme is single-mode (light OR dark). Rather than synthesise the
 * opposite mode, we set both `colors` and `darkColors` to the converted palette
 * so the imported theme renders faithfully no matter where the light/dark toggle
 * sits — `renderedModeFor` still picks the `.dark` class from the real
 * background luminance, so surface-bound UI matches what's on screen.
 */

import { ensureContrast, luminance, mix, normalizeHex, readableOn } from './color'
import type { DesktopTheme, DesktopThemeColors } from './types'

// Section headers / sidebar labels render in --theme-primary directly on the
// sidebar surface as small (~10px) uppercase text, so the accent has to clear
// WCAG AA for normal text (4.5:1) or it's unreadable — the "invisible purple
// label" case. Imported accents below this get nudged lighter/darker.
const ACCENT_MIN_CONTRAST = 4.5

/** The shape of a VS Code `*-color-theme.json` (only the fields we read). */
export interface VscodeColorTheme {
  name?: string
  type?: string
  /** Relative path to a base theme this one extends. We don't follow it. */
  include?: string
  colors?: Record<string, unknown>
  tokenColors?: unknown
}

export interface ConvertOptions {
  /** Stable id (slug). Defaults to a slug of `raw.name`. */
  slug?: string
  /** Display label. Defaults to `raw.name`. */
  label?: string
  /** Shown under the label in the picker (e.g. the marketplace extension id). */
  source?: string
}

export interface ConvertResult {
  theme: DesktopTheme
  /** The source theme's own light/dark (from `type`, else background luminance). */
  mode: 'light' | 'dark'
  /** Workbench keys we wanted but the theme omitted (we derived fallbacks). */
  derived: string[]
}

/** Tolerant slug: lowercase, alnum + dashes, deduped, `vsc-` namespaced. */
export function vscodeThemeSlug(name: string): string {
  const base = name
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 48)

  return `vsc-${base || 'theme'}`
}

/**
 * Parse a VS Code theme file. These ship as JSONC (line/block comments and
 * trailing commas), so a plain `JSON.parse` rejects most real-world files.
 * Strips comments + trailing commas, then parses. Throws on hard syntax errors.
 */
export function parseVscodeTheme(text: string): VscodeColorTheme {
  const stripped = text
    // Block comments.
    .replace(/\/\*[\s\S]*?\*\//g, '')
    // Line comments (not inside strings — naive but fine for theme files).
    .replace(/(^|[^:"'\\])\/\/[^\n\r]*/g, '$1')
    // Trailing commas before } or ].
    .replace(/,(\s*[}\]])/g, '$1')

  const parsed: unknown = JSON.parse(stripped)

  if (!parsed || typeof parsed !== 'object') {
    throw new Error('Theme file is not a JSON object.')
  }

  return parsed as VscodeColorTheme
}

const isDarkType = (raw: VscodeColorTheme, background: string): boolean => {
  const type = (raw.type ?? '').toLowerCase()

  if (type.includes('light')) {
    return false
  }

  if (type === 'dark' || type === 'hc' || type === 'hc-black' || type.includes('dark')) {
    return true
  }

  // No usable `type` — bucket by background luminance.
  return luminance(background) < 0.4
}

/** First normalizable hex among `keys`, composited over `backdrop`. */
const pick = (
  colors: Record<string, unknown>,
  keys: string[],
  backdrop: string
): { key: string; value: string } | null => {
  for (const key of keys) {
    const value = normalizeHex(typeof colors[key] === 'string' ? (colors[key] as string) : null, backdrop)

    if (value) {
      return { key, value }
    }
  }

  return null
}

export function convertVscodeColorTheme(raw: VscodeColorTheme, opts: ConvertOptions = {}): ConvertResult {
  const colors = raw.colors && typeof raw.colors === 'object' ? (raw.colors as Record<string, unknown>) : null

  if (!colors) {
    throw new Error('Theme has no "colors" map — not a VS Code color theme.')
  }

  const derived: string[] = []

  // Background first: it's the backdrop every other token flattens alpha over.
  const backgroundHit = pick(colors, ['editor.background', 'editorPane.background', 'editorGroup.background'], '#000000')
  const dark = isDarkType(raw, backgroundHit?.value ?? '#1e1e1e')
  const background = backgroundHit?.value ?? (dark ? '#1e1e1e' : '#ffffff')

  if (!backgroundHit) {
    derived.push('editor.background')
  }

  // `take` records a derived fallback when the theme omits the key.
  const take = (keys: string[], fallback: string): string => {
    const hit = pick(colors, keys, background)

    if (hit) {
      return hit.value
    }

    derived.push(keys[0])

    return fallback
  }

  const foreground = take(['editor.foreground', 'foreground'], dark ? '#d4d4d4' : '#1f1f1f')

  // Brand accent — the single most load-bearing token. Drives primary buttons,
  // focus rings, the streaming cursor, active-session pills, and sidebar labels.
  // Prefer the saturated "brand" tokens (button / link / badge) over focusBorder,
  // which many themes set to a muted gray — picking it first made imported
  // accents look like the desktop defaults. We enforce contrast below regardless.
  const accentSource = take(
    [
      'button.background',
      'textLink.activeForeground',
      'textLink.foreground',
      'activityBarBadge.background',
      'badge.background',
      'progressBar.background',
      'pickerGroup.foreground',
      'list.highlightForeground',
      'editorLink.activeForeground',
      'focusBorder',
      'tab.activeBorder',
      'statusBarItem.remoteBackground'
    ],
    mix(foreground, background, 0.55)
  )

  const elevated = take(
    ['editorWidget.background', 'dropdown.background', 'menu.background', 'quickInput.background', 'editorSuggestWidget.background'],
    mix(background, foreground, dark ? 0.08 : 0.05)
  )

  const card = take(
    ['sideBarSectionHeader.background', 'tab.inactiveBackground', 'editorGroupHeader.tabsBackground'],
    mix(background, foreground, dark ? 0.04 : 0.025)
  )

  const sidebar = take(['sideBar.background', 'activityBar.background'], mix(background, foreground, dark ? 0.02 : 0.012))

  // The accent labels the sidebar (--theme-primary), so guarantee it reads
  // there — otherwise low-contrast brand colors leave invisible section headers.
  const accent = ensureContrast(accentSource, sidebar, ACCENT_MIN_CONTRAST)

  const border = take(
    ['panel.border', 'editorGroup.border', 'sideBar.border', 'contrastBorder', 'widget.border', 'input.border'],
    mix(background, foreground, dark ? 0.16 : 0.14)
  )

  const input = take(['input.background', 'dropdown.background', 'quickInput.background'], mix(background, foreground, dark ? 0.1 : 0.06))

  const mutedForeground = take(
    ['descriptionForeground', 'editorLineNumber.foreground', 'tab.inactiveForeground', 'disabledForeground'],
    mix(foreground, background, 0.45)
  )

  const destructive = take(
    ['editorError.foreground', 'errorForeground', 'editorOverviewRuler.errorForeground', 'notificationsErrorIcon.foreground'],
    '#e25563'
  )

  const muted = mix(background, foreground, dark ? 0.06 : 0.04)
  const accentSoft = mix(accent, background, dark ? 0.82 : 0.88)
  const secondary = mix(accent, background, dark ? 0.72 : 0.86)

  const palette: DesktopThemeColors = {
    background,
    foreground,
    card,
    cardForeground: foreground,
    muted,
    mutedForeground,
    popover: elevated,
    popoverForeground: foreground,
    primary: accent,
    primaryForeground: readableOn(accent),
    secondary,
    secondaryForeground: foreground,
    accent: accentSoft,
    accentForeground: foreground,
    border,
    input,
    ring: accent,
    midground: accent,
    midgroundForeground: readableOn(accent),
    composerRing: accent,
    destructive,
    destructiveForeground: readableOn(destructive),
    sidebarBackground: sidebar,
    sidebarBorder: border,
    userBubble: mix(card, accent, dark ? 0.18 : 0.12),
    userBubbleBorder: border
  }

  const label = (opts.label ?? raw.name ?? 'VS Code Theme').trim()
  const slug = opts.slug ?? vscodeThemeSlug(label)

  return {
    derived,
    mode: dark ? 'dark' : 'light',
    theme: {
      name: slug,
      label,
      description: opts.source ? `VS Code · ${opts.source}` : 'Imported from VS Code',
      // Single palette in both slots. A lone VS Code theme is one-mode; callers
      // that have both a light and dark variant (a Marketplace extension family)
      // recombine them into proper colors/darkColors via buildThemeFromMarketplace.
      colors: palette,
      darkColors: palette
    }
  }
}
