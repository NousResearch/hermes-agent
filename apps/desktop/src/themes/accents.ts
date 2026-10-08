/**
 * Accent swatches and the ids a stored accent pick is written as.
 *
 * A pick is stored as a swatch id (`mono`, `green`, …) or `custom:#rrggbb`,
 * never as a resolved hex, because one swatch can paint differently per mode:
 * Mono is black on a light surface and white on a dark one.
 */

import { normalizeHex } from './color'

export const NOUS_ACCENT = '#0053fd'

const CUSTOM_PREFIX = 'custom:'

interface Swatch {
  id: string
  name: string
  light: string
  dark: string
}

const SWATCHES: readonly Swatch[] = [
  { id: 'mono', name: 'Mono', light: '#000000', dark: '#ffffff' },
  { id: 'green', name: 'GitHub green', light: '#2ea043', dark: '#2ea043' },
  { id: 'cyan', name: 'Cyber cyan', light: '#00d5ff', dark: '#00d5ff' },
  { id: 'nous', name: 'Nous blue', light: NOUS_ACCENT, dark: NOUS_ACCENT },
  { id: 'violet', name: 'Ultraviolet', light: '#8a2be2', dark: '#8a2be2' },
  { id: 'pink', name: 'Barbie pink', light: '#e0218a', dark: '#e0218a' },
  { id: 'red', name: 'Electric red', light: '#ff073a', dark: '#ff073a' },
  { id: 'orange', name: 'Safety orange', light: '#ff6a00', dark: '#ff6a00' }
]

export interface AccentSwatchSpec {
  hex: string
  id: string
  name: string
}

export const accentsFor = (dark: boolean): AccentSwatchSpec[] =>
  SWATCHES.map(({ id, name, light, dark: onDark }) => ({ hex: dark ? onDark : light, id, name }))

export const customAccentId = (hex: string): string => `${CUSTOM_PREFIX}${hex}`

/**
 * The id worth storing for a pick, or `null` for the theme's own accent.
 * Nous blue is the default skin's accent, so it is stored as `null` rather
 * than pinned; an unknown id or a malformed custom hex also falls to `null`.
 */
export function normalizeAccentId(value: null | string): null | string {
  if (value?.startsWith(CUSTOM_PREFIX)) {
    const hex = normalizeHex(value.slice(CUSTOM_PREFIX.length))

    return hex && hex !== NOUS_ACCENT ? customAccentId(hex) : null
  }

  return value !== 'nous' && SWATCHES.some(swatch => swatch.id === value) ? value : null
}

/** The hex a stored id paints in the given mode; `null` keeps the theme's accent. */
export function resolveAccent(value: null | string, dark: boolean): null | string {
  const id = normalizeAccentId(value)

  if (id?.startsWith(CUSTOM_PREFIX)) {
    return id.slice(CUSTOM_PREFIX.length)
  }

  return accentsFor(dark).find(swatch => swatch.id === id)?.hex ?? null
}
