export interface WallpaperPalette {
  accent: string
  dominant: string
}

export function sanitizeWallpaperPalette(value: unknown): WallpaperPalette | null {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return null
  }

  const candidate = value as Record<string, unknown>
  const valid = (color: unknown): color is string => typeof color === 'string' && /^#[0-9a-f]{6}$/i.test(color)

  return valid(candidate.accent) && valid(candidate.dominant)
    ? { accent: candidate.accent.toLowerCase(), dominant: candidate.dominant.toLowerCase() }
    : null
}
