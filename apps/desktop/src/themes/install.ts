/**
 * Install desktop themes from external sources.
 *
 * The heavy lifting (network + .vsix unzip) lives in the Electron main process
 * (`electron/vscode-marketplace.cjs`), reached via `window.hermesDesktop.themes`.
 * Main hands back the raw theme JSON; we parse + convert + persist here so the
 * conversion stays in one unit-testable place.
 */

import type { DesktopMarketplaceThemeResult } from '@/global'

import type { DesktopTheme } from './types'
import { installUserTheme } from './user-themes'
import { convertVscodeColorTheme, parseVscodeTheme, vscodeThemeSlug } from './vscode'

/** A `publisher.extension` id, e.g. `dracula-theme.theme-dracula`. */
export const MARKETPLACE_ID_RE = /^[\w-]+\.[\w-]+$/

/** Parse + convert + persist a pasted VS Code theme JSON. */
export function installVscodeThemeFromText(
  text: string,
  opts?: { label?: string; source?: string }
): DesktopTheme {
  const raw = parseVscodeTheme(text)
  const { theme } = convertVscodeColorTheme(raw, opts)

  return installUserTheme(theme)
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

    return { mode, palette: theme.colors }
  })

  const fallback = variants[0].palette
  const light = variants.find(variant => variant.mode === 'light')?.palette
  const dark = variants.find(variant => variant.mode === 'dark')?.palette

  return {
    name: vscodeThemeSlug(result.displayName),
    label: result.displayName,
    description: `VS Code · ${result.extensionId}`,
    colors: light ?? dark ?? fallback,
    darkColors: dark ?? light ?? fallback
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
