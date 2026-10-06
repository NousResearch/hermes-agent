/**
 * Install desktop themes from external sources.
 *
 * The heavy lifting (network + .vsix unzip) lives in the Electron main process
 * (`electron/vscode-marketplace.ts`), reached via `window.hermesDesktop.themes`.
 * Main hands back the raw theme JSON; we parse + convert + persist here so the
 * conversion stays in one unit-testable place.
 */

import type { DesktopMarketplaceThemeResult } from '@/global'

import type { DesktopTerminalPalette, DesktopTheme, DesktopThemeColors, DesktopThemeVariant } from './types'
import { installUserTheme } from './user-themes'
import { convertVscodeColorTheme, parseVscodeTheme, vscodeThemeSlug } from './vscode'

/** A `publisher.extension` id, e.g. `dracula-theme.theme-dracula`. */
export const MARKETPLACE_ID_RE = /^[\w-]+\.[\w-]+$/

/** One contributed palette, before it is folded into the family. */
interface ContributedPalette {
  mode: 'light' | 'dark'
  label: string
  palette: DesktopThemeColors
  terminal?: DesktopTerminalPalette
}

/** Tolerant variant slug: lowercase, alnum + dashes, no `vsc-` prefix. */
const variantSlug = (label: string): string =>
  label
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 48) || 'variant'

/**
 * Drop the shared label prefix so a picker shows "Wave", not "Kanagawa Wave".
 * Whole leading words only, and never to the point of emptying a label.
 */
function stripCommonPrefix(labels: string[]): string[] {
  const tokens = labels.map(label => label.trim().split(/\s+/))
  const first = tokens[0] ?? []
  let shared = 0

  for (let i = 0; i < first.length; i++) {
    const token = first[i].toLowerCase()

    if (tokens.every(parts => parts[i]?.toLowerCase() === token)) {
      shared = i + 1
    } else {
      break
    }
  }

  if (shared === 0) {
    return labels
  }

  const stripped = tokens.map(parts => parts.slice(shared).join(' '))

  return stripped.some(label => !label) ? labels : stripped
}

/**
 * Keep a family's same-mode flavors (Kanagawa's Wave/Dragon/Lotus) as pickable
 * variants. A plain light+dark pair needs none — the mode toggle covers it.
 */
function buildVariants(palettes: ContributedPalette[]): DesktopThemeVariant[] | undefined {
  const lights = palettes.filter(palette => palette.mode === 'light').length
  const darks = palettes.filter(palette => palette.mode === 'dark').length

  if (lights <= 1 && darks <= 1) {
    return undefined
  }

  const labels = stripCommonPrefix(palettes.map(palette => palette.label))
  const seen = new Set<string>()

  const variants = palettes.map((palette, index): DesktopThemeVariant => {
    let name = variantSlug(labels[index])

    while (seen.has(name)) {
      name = `${name}-${index}`
    }

    seen.add(name)

    return {
      name,
      label: labels[index],
      mode: palette.mode,
      colors: palette.palette,
      ...(palette.terminal ? { terminal: palette.terminal } : {})
    }
  })

  return variants.length > 1 ? variants : undefined
}

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
 *
 * Extensions with several palettes in ONE mode (Kanagawa Flavors) can't fit
 * the pair, so those extras ride along as `variants` for the card picker.
 */
export function buildThemeFromMarketplace(result: DesktopMarketplaceThemeResult): DesktopTheme {
  if (!result.themes.length) {
    throw new Error(`"${result.extensionId}" does not contribute any color themes.`)
  }

  const palettes: ContributedPalette[] = result.themes.map(file => {
    const raw = parseVscodeTheme(file.contents)
    const label = file.label || raw.name || result.displayName
    const { mode, theme } = convertVscodeColorTheme(raw, { label, source: result.extensionId })

    return { mode, label, palette: theme.colors, terminal: theme.terminal }
  })

  const fallback = palettes[0]
  const light = palettes.find(palette => palette.mode === 'light') ?? fallback
  const dark = palettes.find(palette => palette.mode === 'dark') ?? fallback

  // The terminal ANSI palette tracks the painted variant the same way colors do
  // (light → terminal, dark → darkTerminal); each falls back to the other so a
  // single-variant import still themes the terminal in both modes.
  const terminal = light.terminal ?? dark.terminal
  const darkTerminal = dark.terminal ?? light.terminal
  const variants = buildVariants(palettes)

  return {
    name: vscodeThemeSlug(result.displayName),
    label: result.displayName,
    description: `VS Code · ${result.extensionId}`,
    colors: light.palette,
    darkColors: dark.palette,
    ...(terminal ? { terminal } : {}),
    ...(darkTerminal ? { darkTerminal } : {}),
    ...(variants ? { variants } : {})
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
