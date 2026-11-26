/**
 * Accent swatches and the ids the agent and the picker use for them.
 *
 * A swatch can paint differently per surface (Mono is black on light, white on
 * dark), so callers resolve an id against the rendered mode at the moment of
 * the pick. `custom:#rrggbb` names a colour that is not a swatch.
 */

export const NOUS_ACCENT = '#0053fd'

export const CUSTOM_ACCENT_PREFIX = 'custom:'

interface Swatch {
  dark: string
  id: string
  light: string
  name: string
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

export const ACCENT_SWATCH_IDS: readonly string[] = SWATCHES.map(swatch => swatch.id)

export interface AccentSwatchSpec {
  hex: string
  id: string
  name: string
}

export const accentsFor = (dark: boolean): AccentSwatchSpec[] =>
  SWATCHES.map(({ dark: onDark, id, light, name }) => ({ hex: dark ? onDark : light, id, name }))

/** A pick the agent or a card names: its hex on this surface and the id reported back (`null` = the theme's own). */
export interface ResolvedAccent {
  hex: null | string
  id: null | string
}

/**
 * Resolve `theme`, a swatch id, `#rrggbb` or `custom:#rrggbb` to what gets painted. Nous blue is the default
 * skin's accent, so it resolves to the theme's own (`null`), the same as `theme`. Unknown input → `undefined`.
 */
export function resolveAccentPick(value: string, dark: boolean): ResolvedAccent | undefined {
  if (value === 'theme' || value === 'nous') {
    return { hex: null, id: null }
  }

  const swatch = accentsFor(dark).find(candidate => candidate.id === value)

  if (swatch) {
    return { hex: swatch.hex, id: swatch.id }
  }

  const raw = value.startsWith(CUSTOM_ACCENT_PREFIX) ? value.slice(CUSTOM_ACCENT_PREFIX.length) : value

  if (!/^#[0-9a-f]{6}$/i.test(raw)) {
    return undefined
  }

  const hex = raw.toLowerCase()

  return hex === NOUS_ACCENT ? { hex: null, id: null } : { hex, id: `${CUSTOM_ACCENT_PREFIX}${hex}` }
}

/** The name a settled card shows for a reported accent id. */
export function accentName(id: string): string {
  return SWATCHES.find(swatch => swatch.id === id)?.name ?? id.replace(CUSTOM_ACCENT_PREFIX, '')
}
