/**
 * Install desktop themes from external sources.
 *
 * The heavy lifting (network + .vsix unzip) lives in the Electron main process
 * (`electron/vscode-marketplace.cjs`), reached via `window.hermesDesktop.themes`.
 * Main hands back the raw theme JSON; we parse + convert + persist here so the
 * conversion stays in one unit-testable place.
 */

import type { DesktopMarketplaceThemeResult } from '@/global'

import { BUILTIN_THEMES } from './presets'
import type { DesktopTheme } from './types'
import { installUserTheme } from './user-themes'
import { convertVscodeColorTheme, parseVscodeTheme, vscodeThemeSlug } from './vscode'

/** A `publisher.extension` id, e.g. `dracula-theme.theme-dracula`. */
export const MARKETPLACE_ID_RE = /^[\w-]+\.[\w-]+$/

/** Parse + convert + persist a pasted VS Code theme JSON. */
export function installVscodeThemeFromText(text: string, opts?: { label?: string; source?: string }): DesktopTheme {
  const raw = parseVscodeTheme(text)
  const { theme } = convertVscodeColorTheme(raw, opts)

  return installUserTheme(theme)
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value && typeof value === 'object' && !Array.isArray(value))
}

function themePayloads(parsed: unknown): unknown[] {
  if (isRecord(parsed) && Array.isArray(parsed.themes)) {
    return parsed.themes
  }

  if (isRecord(parsed) && typeof parsed.name === 'string' && isRecord(parsed.colors)) {
    return [parsed]
  }

  throw new Error('Expected a Hermes theme JSON object or { "themes": [...] } theme pack.')
}

function validateHermesTheme(value: unknown, index: number): DesktopTheme {
  if (!isRecord(value)) {
    throw new Error(`Theme ${index + 1} must be a JSON object.`)
  }

  const name = value.name
  const label = value.label
  const description = value.description
  const colors = value.colors

  if (typeof name !== 'string' || !name.trim()) {
    throw new Error(`Theme ${index + 1} is missing a valid name.`)
  }

  if (typeof label !== 'string' || !label.trim()) {
    throw new Error(`Theme "${name}" is missing a valid label.`)
  }

  if (typeof description !== 'string') {
    throw new Error(`Theme "${name}" is missing a valid description.`)
  }

  if (BUILTIN_THEMES[name]) {
    throw new Error(`"${name}" collides with a built-in theme.`)
  }

  if (!isRecord(colors)) {
    throw new Error(`Theme "${name}" is missing required colors.`)
  }

  for (const key of ['background', 'foreground', 'primary'] as const) {
    if (typeof colors[key] !== 'string') {
      throw new Error(`Theme "${name}" is missing required color "${key}".`)
    }
  }

  return value as unknown as DesktopTheme
}

/** Parse and install Hermes-native DesktopTheme JSON (single theme or pack). */
export function installHermesThemeFromText(text: string): DesktopTheme[] {
  let parsed: unknown

  try {
    parsed = JSON.parse(text)
  } catch {
    throw new Error('Theme file is not valid JSON.')
  }

  const themes = themePayloads(parsed).map(validateHermesTheme)

  return themes.map(theme => installUserTheme(theme))
}

/**
 * Fold every color theme an extension contributes into ONE desktop theme family.
 *
 * Many extensions ship a light *and* a dark variant (GitHub, Solarized, Winter
 * is Coming…). Rather than install them as separate flat entries — which made
 * the light/dark toggle a no-op and let "install in dark mode" land on the light
 * variant — we map the first light variant onto `colors` and the first dark
 * variant onto `darkColors`. The result is a single picker entry whose light/dark
 * toggle switches between the real variants. A single-variant extension fills
 * both slots with its one palette (the toggle is a no-op, as it must be).
 */
export function buildThemeFromMarketplace(result: DesktopMarketplaceThemeResult): DesktopTheme {
  if (!result.themes.length) {
    throw new Error(`"${result.extensionId}" does not contribute any color themes.`)
  }

  const variants = result.themes.map(file => {
    const raw = parseVscodeTheme(file.contents)
    const label = file.label || raw.name || result.displayName
    const { mode, theme } = convertVscodeColorTheme(raw, { label, source: result.extensionId })

    return { mode, palette: theme.colors, terminal: theme.terminal }
  })

  const fallback = variants[0]
  const light = variants.find(variant => variant.mode === 'light') ?? fallback
  const dark = variants.find(variant => variant.mode === 'dark') ?? fallback

  // The terminal ANSI palette tracks the painted variant the same way colors do
  // (light → terminal, dark → darkTerminal); each falls back to the other so a
  // single-variant import still themes the terminal in both modes.
  const terminal = light.terminal ?? dark.terminal
  const darkTerminal = dark.terminal ?? light.terminal

  return {
    name: vscodeThemeSlug(result.displayName),
    label: result.displayName,
    description: `VS Code · ${result.extensionId}`,
    colors: light.palette,
    darkColors: dark.palette,
    ...(terminal ? { terminal } : {}),
    ...(darkTerminal ? { darkTerminal } : {})
  }
}

/**
 * Download a Marketplace extension and install the theme family it contributes
 * (see `buildThemeFromMarketplace`). Returns the single installed theme.
 */
export async function installVscodeThemeFromMarketplace(id: string): Promise<DesktopTheme> {
  const trimmed = id.trim()

  if (!MARKETPLACE_ID_RE.test(trimmed)) {
    throw new Error('Expected a Marketplace id like "publisher.extension".')
  }

  const api = window.hermesDesktop?.themes

  if (!api?.fetchMarketplace) {
    throw new Error('Marketplace install is only available in the desktop app.')
  }

  const result = await api.fetchMarketplace(trimmed)

  return installUserTheme(buildThemeFromMarketplace(result))
}
