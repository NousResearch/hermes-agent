// The redesign draws agents in Figma's five primaries (Var1 147:7470 solid tiles; 895:17703 tiles at
// 25% with a hue ring), not the classic hashed hues. A profile's colour (its pick or its hash, from
// profile-color.ts, which classic keeps using) snaps to the nearest primary by hue. Each primary has
// a solid shade that keeps the white robot glyph at >= 3.5:1, and the Figma hue for tints and rings.

export interface PortalAgentSwatch {
  /** Figma's primary, for tints and rings (895:17703). */
  hue: string
  /** The solid tile under the white glyph (Var1 147:7470). */
  solid: string
}

const PRIMARIES: readonly { at: number; swatch: PortalAgentSwatch }[] = [
  { at: 0, swatch: { hue: '#f20000', solid: '#f20000' } },
  { at: 48, swatch: { hue: '#f2ca00', solid: '#8a6f00' } },
  { at: 160, swatch: { hue: '#00f279', solid: '#0e9a6b' } },
  { at: 240, swatch: { hue: '#0000f2', solid: '#0000f2' } },
  { at: 271, swatch: { hue: '#a252ee', solid: '#9340e6' } }
]

/** The default agent, and any colour without a readable hue: the brand primary. */
const BRAND = PRIMARIES[3]!.swatch

/** The agent colour picker's swatches in Figma's order (895:9675): brand first, then by hue. */
export const PORTAL_AGENT_HUES: readonly string[] = [
  BRAND,
  ...PRIMARIES.filter(p => p.swatch !== BRAND).map(p => p.swatch)
].map(swatch => swatch.hue)

function hexHue(hex: string): null | number {
  const value = hex.length === 4 ? hex.replace(/(\w)/g, '$1$1').slice(1) : hex.slice(1)

  if (!/^[\da-f]{6}$/i.test(value)) {
    return null
  }

  const [r, g, b] = [0, 2, 4].map(i => Number.parseInt(value.slice(i, i + 2), 16) / 255) as [number, number, number]
  const max = Math.max(r, g, b)
  const delta = max - Math.min(r, g, b)

  if (!delta) {
    return null
  }

  const sector = max === r ? ((g - b) / delta + 6) % 6 : max === g ? (b - r) / delta + 2 : (r - g) / delta + 4

  return sector * 60
}

function colorHue(color: string): null | number {
  const hsl = /^hsla?\(\s*(-?[\d.]+)/i.exec(color.trim())

  if (hsl) {
    return ((Number(hsl[1]) % 360) + 360) % 360
  }

  return color.trim().startsWith('#') ? hexHue(color.trim()) : null
}

const hueDistance = (a: number, b: number) => Math.min(Math.abs(a - b), 360 - Math.abs(a - b))

/** The redesign's swatch for a profile colour (`resolveProfileColor`); null is the default agent. */
export function portalAgentSwatch(color: null | string | undefined): PortalAgentSwatch {
  const hue = color ? colorHue(color) : null

  if (hue === null) {
    return BRAND
  }

  return PRIMARIES.reduce((best, entry) => (hueDistance(entry.at, hue) < hueDistance(best.at, hue) ? entry : best))
    .swatch
}
