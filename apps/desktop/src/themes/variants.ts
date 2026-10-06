/**
 * Theme variants — alternate same-mode palettes inside a theme family.
 *
 * `install.ts` keeps a family's same-mode flavors on `variants`; the picker on
 * the active Appearance card reads and writes the choice here. Selection is per
 * profile + theme, like every other appearance choice.
 */

import { readJson, writeJson } from '@/lib/storage'

import type { DesktopTheme, DesktopThemeVariant } from './types'

const VARIANTS_KEY = 'hermes-desktop-theme-variants-v1'

/** Exposed so the theme context can also repaint on a peer window's pick. */
export const THEME_VARIANTS_KEY = VARIANTS_KEY

/** `{ [profile]: { [themeName]: variantName } }`. */
type VariantSelections = Record<string, Record<string, string>>

function readSelections(): VariantSelections {
  const parsed = readJson<VariantSelections>(VARIANTS_KEY)

  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
    return {}
  }

  const out: VariantSelections = {}

  for (const [profile, themes] of Object.entries(parsed)) {
    if (!themes || typeof themes !== 'object' || Array.isArray(themes)) {
      continue
    }

    const picks: Record<string, string> = {}

    for (const [theme, variant] of Object.entries(themes)) {
      if (typeof variant === 'string') {
        picks[theme] = variant
      }
    }

    out[profile] = picks
  }

  return out
}

/** The stored variant pick for a profile + theme, un-normalized. */
export function storedThemeVariant(profile: string, theme: string): string | null {
  return readSelections()[profile]?.[theme] ?? null
}

/** Persist (or clear, with `null`) a profile's variant pick for a theme. */
export function persistThemeVariant(profile: string, theme: string, variant: string | null): void {
  const all = readSelections()
  const picks = { ...(all[profile] ?? {}) }

  if (variant === null) {
    delete picks[theme]
  } else {
    picks[theme] = variant
  }

  const next: VariantSelections = { ...all }

  if (Object.keys(picks).length > 0) {
    next[profile] = picks
  } else {
    delete next[profile]
  }

  writeJson(VARIANTS_KEY, next)
}

/** The variants a family offers for one mode, in contribution order. */
export function variantsForMode(theme: DesktopTheme | undefined, mode: 'light' | 'dark'): DesktopThemeVariant[] {
  const variants = theme?.variants

  return Array.isArray(variants) ? variants.filter(variant => variant.mode === mode) : []
}

/**
 * The variant in force for `mode`: the stored pick when it belongs to that
 * mode, else the first offered.
 */
export function activeVariantName(
  theme: DesktopTheme | undefined,
  mode: 'light' | 'dark',
  selected: string | null
): string | null {
  const options = variantsForMode(theme, mode)

  if (selected && options.some(variant => variant.name === selected)) {
    return selected
  }

  return options[0]?.name ?? null
}

/** The variant palette in force for `mode`, or undefined when the family has none. */
export function variantForMode(
  theme: DesktopTheme | undefined,
  mode: 'light' | 'dark',
  selected: string | null
): DesktopThemeVariant | undefined {
  const name = activeVariantName(theme, mode, selected)

  return name ? variantsForMode(theme, mode).find(variant => variant.name === name) : undefined
}
