/**
 * Install desktop themes from external sources.
 *
 * The heavy lifting (network + .vsix unzip) lives in the Electron main process
 * (`electron/vscode-marketplace.ts`), reached via `window.hermesDesktop.themes`.
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
export function installVscodeThemeFromText(text: string, opts?: { label?: string; source?: string }): DesktopTheme {
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
 * When an extension contributes 3+ themes (e.g., Catppuccin with 4 flavors),
 * install each as its own independent theme so users can pick the specific
 * variant they want (Mocha, Macchiato, etc.) rather than folding them all
 * into one family that forces the light/dark toggle to switch variants.
 *
 * Returns the first dark variant (most users installing a dark theme want the
 * fullest dark), falling back to the first light variant, then the first entry.
 */
function installMultiVariantThemes(result: DesktopMarketplaceThemeResult): DesktopTheme {
  const installed: DesktopTheme[] = []
  let firstDark: DesktopTheme | undefined
  let firstLight: DesktopTheme | undefined

  for (const file of result.themes) {
    const raw = parseVscodeTheme(file.contents)
    const label = file.label || raw.name || result.displayName
    const { mode, theme } = convertVscodeColorTheme(raw, { label, source: result.extensionId })

    // Each variant is a standalone single-mode theme.  Omitting darkColors means
    // getBaseColors returns `colors` for *both* light and dark — the light/dark
    // toggle stays a no-op on the palette, which is correct since each variant
    // IS a single mode and shouldn't jump to another variant when toggled.
    const standalone: DesktopTheme = {
      ...theme,
      label,
      darkColors: undefined
    }

    const stored = installUserTheme(standalone)
    installed.push(stored)

    if (mode === 'dark' && !firstDark) {
      firstDark = stored
    }

    if (mode === 'light' && !firstLight) {
      firstLight = stored
    }
  }

  // Activate the first dark variant so a Catppuccin install lands on Mocha.
  return firstDark ?? firstLight ?? installed[0]
}

/**
 * Download a Marketplace extension and install the theme(s) it contributes.
 *
 * - 1–2 themes (common 1-light-1-dark pair, e.g. Solarized, GitHub):
 *   fold into a single family with `colors` / `darkColors` so the light/dark
 *   toggle switches between the real variants.  `buildThemeFromMarketplace`
 *   handles this.
 * - 3+ themes (multi-variant like Catppuccin's 4 flavors): install each as
 *   its own independent picker entry via `installMultiVariantThemes`.
 *
 * Returns the single theme to activate.
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

  if (result.themes.length >= 3) {
    return installMultiVariantThemes(result)
  }

  return installUserTheme(buildThemeFromMarketplace(result))
}
