import { atom } from 'nanostores'

import { contrastRatio, ensureContrast, mix, normalizeHex, readableOn } from './color'
import { DEFAULT_TYPOGRAPHY } from './presets'
import type { DesktopTheme, DesktopThemeColors } from './types'
import { installUserTheme, removeUserTheme, resolveTheme } from './user-themes'

const CUSTOM_THEMES_KEY = 'hermes-desktop-custom-theme-definitions-v1'
const CUSTOM_THEME_VERSION = 1
const DEFAULT_CONTRAST = 38

export interface CustomThemePaletteSeed {
  accent: string
  background: string
  foreground: string
  contrast: number
}

/** Default authoring palette for a new custom theme's Light mode. */
export const DEFAULT_CUSTOM_LIGHT_PALETTE: Readonly<CustomThemePaletteSeed> = {
  accent: '#cc7d5e',
  background: '#f9f9f7',
  foreground: '#2d2d2b',
  contrast: DEFAULT_CONTRAST
}

/** Default authoring palette for a new custom theme's Dark mode. */
export const DEFAULT_CUSTOM_DARK_PALETTE: Readonly<CustomThemePaletteSeed> = {
  accent: '#cc7d5e',
  background: '#2d2d2b',
  foreground: '#f9f9f7',
  contrast: DEFAULT_CONTRAST
}

export interface CustomThemeDefinition {
  version: typeof CUSTOM_THEME_VERSION
  name: string
  label: string
  baseTheme: string
  light: CustomThemePaletteSeed
  dark: CustomThemePaletteSeed
  fontSans: string
  fontMono: string
  translucentSidebar: boolean
}

/**
 * Restore the product's authoring palettes without discarding the user's
 * theme identity, typography, or sidebar preference.
 */
export function resetCustomThemePalettes(definition: CustomThemeDefinition): CustomThemeDefinition {
  return {
    ...definition,
    light: { ...DEFAULT_CUSTOM_LIGHT_PALETTE },
    dark: { ...DEFAULT_CUSTOM_DARK_PALETTE }
  }
}

export interface ThemePreview {
  theme: DesktopTheme
  mode: 'light' | 'dark'
}

interface StoredCustomThemes {
  version: typeof CUSTOM_THEME_VERSION
  themes: Record<string, CustomThemeDefinition>
}

const isPaletteSeed = (value: unknown): value is CustomThemePaletteSeed => {
  if (!value || typeof value !== 'object') {
    return false
  }

  const seed = value as Partial<CustomThemePaletteSeed>

  return (
    normalizeHex(seed.accent) !== null &&
    normalizeHex(seed.background) !== null &&
    normalizeHex(seed.foreground) !== null &&
    typeof seed.contrast === 'number' &&
    Number.isFinite(seed.contrast)
  )
}

const isDefinition = (value: unknown): value is CustomThemeDefinition => {
  if (!value || typeof value !== 'object') {
    return false
  }

  const definition = value as Partial<CustomThemeDefinition>

  return (
    definition.version === CUSTOM_THEME_VERSION &&
    typeof definition.name === 'string' &&
    definition.name.length > 0 &&
    typeof definition.label === 'string' &&
    typeof definition.baseTheme === 'string' &&
    typeof definition.fontSans === 'string' &&
    typeof definition.fontMono === 'string' &&
    typeof definition.translucentSidebar === 'boolean' &&
    isPaletteSeed(definition.light) &&
    isPaletteSeed(definition.dark)
  )
}

function readStored(): Record<string, CustomThemeDefinition> {
  try {
    const raw = window.localStorage.getItem(CUSTOM_THEMES_KEY)

    if (!raw) {
      return {}
    }

    const parsed: unknown = JSON.parse(raw)

    if (!parsed || typeof parsed !== 'object') {
      return {}
    }

    const stored = parsed as Partial<StoredCustomThemes>

    if (stored.version !== CUSTOM_THEME_VERSION || !stored.themes || typeof stored.themes !== 'object') {
      return {}
    }

    return Object.fromEntries(
      Object.entries(stored.themes).filter(([name, definition]) => name === definition?.name && isDefinition(definition))
    )
  } catch {
    return {}
  }
}

function persist(themes: Record<string, CustomThemeDefinition>): void {
  try {
    const stored: StoredCustomThemes = { version: CUSTOM_THEME_VERSION, themes }
    window.localStorage.setItem(CUSTOM_THEMES_KEY, JSON.stringify(stored))
  } catch {
    // Best-effort: theme editing should still work in restricted storage.
  }
}

const normalizedColor = (value: string, backdrop?: string): string => {
  const color = normalizeHex(value, backdrop)

  if (!color) {
    throw new Error(`Invalid theme color: ${value}`)
  }

  return color
}

const normalizedSeed = (seed: CustomThemePaletteSeed): CustomThemePaletteSeed => {
  const background = normalizedColor(seed.background)

  return {
    accent: normalizedColor(seed.accent, background),
    background,
    foreground: normalizedColor(seed.foreground, background),
    contrast: Math.round(Math.min(100, Math.max(0, seed.contrast)))
  }
}

/**
 * Derive the existing desktop tokens from the small set exposed by the editor.
 * Contrast changes only the distance between surfaces and their borders; it
 * never changes the chosen background or foreground.
 */
export function deriveCustomThemeColors(
  input: CustomThemePaletteSeed,
  translucentSidebar: boolean
): DesktopThemeColors {
  const seed = normalizedSeed(input)
  const strength = seed.contrast / 100
  const background = seed.background
  const foreground = ensureContrast(seed.foreground, background, 4.5)
  const card = mix(background, foreground, 0.012 + strength * 0.038)
  const popover = mix(background, foreground, 0.02 + strength * 0.05)
  const muted = mix(background, foreground, 0.035 + strength * 0.075)
  const border = mix(background, foreground, 0.1 + strength * 0.23)
  const inputSurface = mix(background, foreground, 0.065 + strength * 0.19)
  const secondary = mix(background, seed.accent, 0.055 + strength * 0.105)
  const accentSurface = mix(background, seed.accent, 0.08 + strength * 0.16)
  const sidebar = mix(background, seed.accent, 0.025 + strength * 0.055)
  const primary = ensureContrast(ensureContrast(seed.accent, background, 3), sidebar, 3)

  const sidebarBackground = translucentSidebar
    ? `color-mix(in srgb, ${sidebar} 82%, transparent)`
    : sidebar

  return {
    background,
    foreground,
    card,
    cardForeground: ensureContrast(foreground, card, 4.5),
    muted,
    mutedForeground: ensureContrast(mix(foreground, background, 0.32), muted, 4.5),
    popover,
    popoverForeground: ensureContrast(foreground, popover, 4.5),
    primary,
    primaryForeground: readableOn(primary),
    secondary,
    secondaryForeground: ensureContrast(foreground, secondary, 4.5),
    accent: accentSurface,
    accentForeground: ensureContrast(foreground, accentSurface, 4.5),
    border,
    input: inputSurface,
    ring: primary,
    midground: primary,
    midgroundForeground: readableOn(primary),
    composerRing: primary,
    destructive: ensureContrast('#c74444', background, 3),
    destructiveForeground: '#ffffff',
    sidebarBackground,
    sidebarBorder: border,
    userBubble: secondary,
    userBubbleBorder: border
  }
}

export function buildCustomTheme(definition: CustomThemeDefinition): DesktopTheme {
  if (!definition.label.trim() || !definition.fontSans.trim() || !definition.fontMono.trim()) {
    throw new Error('Theme name and fonts are required.')
  }

  const light = normalizedSeed(definition.light)
  const dark = normalizedSeed(definition.dark)

  return {
    name: definition.name,
    label: definition.label.trim(),
    description: `Custom theme · ${definition.baseTheme}`,
    colors: deriveCustomThemeColors(light, definition.translucentSidebar),
    darkColors: deriveCustomThemeColors(dark, definition.translucentSidebar),
    typography: {
      fontSans: definition.fontSans.trim(),
      fontMono: definition.fontMono.trim()
    }
  }
}

const paletteSeedFrom = (colors: DesktopThemeColors): CustomThemePaletteSeed => ({
  accent: normalizeHex(colors.midground ?? colors.ring ?? colors.primary, colors.background) ?? '#0053fd',
  background: normalizeHex(colors.background) ?? '#ffffff',
  foreground: normalizeHex(colors.foreground, colors.background) ?? '#161616',
  contrast: DEFAULT_CONTRAST
})

const slugFor = (label: string): string =>
  label
    .normalize('NFKD')
    .toLowerCase()
    .replace(/[^\p{Letter}\p{Number}]+/gu, '-')
    .replace(/^-+|-+$/g, '') || 'theme'

export function uniqueCustomThemeName(label: string, currentName?: string): string {
  const base = `custom-${slugFor(label)}`
  let candidate = base
  let suffix = 2

  while (candidate !== currentName && (resolveTheme(candidate) || $customThemeDefinitions.get()[candidate])) {
    candidate = `${base}-${suffix}`
    suffix += 1
  }

  return candidate
}

export function createCustomThemeDefinition({
  source,
  lightColors,
  darkColors,
  label = `${source.label} Custom`
}: {
  source: DesktopTheme
  lightColors: DesktopThemeColors
  darkColors: DesktopThemeColors
  label?: string
}): CustomThemeDefinition {
  return {
    version: CUSTOM_THEME_VERSION,
    name: uniqueCustomThemeName(label),
    label,
    baseTheme: source.name,
    // "Create custom theme" is a copy operation regardless of where the
    // active theme came from. Reset remains the explicit way back to Hermes'
    // warm neutral authoring defaults.
    light: paletteSeedFrom(lightColors),
    dark: paletteSeedFrom(darkColors),
    fontSans: source.typography?.fontSans ?? DEFAULT_TYPOGRAPHY.fontSans,
    fontMono: source.typography?.fontMono ?? DEFAULT_TYPOGRAPHY.fontMono,
    translucentSidebar: false
  }
}

export const $customThemeDefinitions = atom<Record<string, CustomThemeDefinition>>(
  typeof window === 'undefined' ? {} : readStored()
)

/** Ephemeral authoring state. This atom is deliberately never persisted. */
export const $themePreview = atom<ThemePreview | null>(null)

export function getCustomThemeDefinition(name: string): CustomThemeDefinition | undefined {
  return $customThemeDefinitions.get()[name]
}

export function isCustomTheme(name: string): boolean {
  return Boolean(getCustomThemeDefinition(name))
}

export function saveCustomTheme(definition: CustomThemeDefinition): DesktopTheme {
  const normalized: CustomThemeDefinition = {
    ...definition,
    label: definition.label.trim(),
    light: normalizedSeed(definition.light),
    dark: normalizedSeed(definition.dark),
    fontSans: definition.fontSans.trim(),
    fontMono: definition.fontMono.trim()
  }

  const theme = buildCustomTheme(normalized)

  installUserTheme(theme)

  const next = { ...$customThemeDefinitions.get(), [normalized.name]: normalized }
  $customThemeDefinitions.set(next)
  persist(next)

  return theme
}

export function removeCustomTheme(name: string): void {
  const current = $customThemeDefinitions.get()

  if (current[name]) {
    const next = { ...current }
    delete next[name]
    $customThemeDefinitions.set(next)
    persist(next)
  }

  removeUserTheme(name)

  if ($themePreview.get()?.theme.name === name) {
    $themePreview.set(null)
  }
}

export function customThemeContrast(theme: DesktopTheme, mode: 'light' | 'dark'): {
  accent: number
  text: number
} {
  const colors = mode === 'dark' ? (theme.darkColors ?? theme.colors) : theme.colors
  const sidebar = normalizeHex(colors.sidebarBackground, colors.background) ?? colors.background

  return {
    accent: Math.min(contrastRatio(colors.primary, colors.background), contrastRatio(colors.primary, sidebar)),
    text: contrastRatio(colors.foreground, colors.background)
  }
}
