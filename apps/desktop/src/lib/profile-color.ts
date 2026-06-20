// Deterministic per-profile color so a profile is glanceable across the app
// (the sidebar profile rail). The default/root profile has no color — named
// profiles get a stable hue derived from the name, so the same profile always
// reads the same color without persisting anything.

const PROFILE_TAG_SATURATION = 68
const PROFILE_TAG_LIGHTNESS = 58

function hashString(value: string): number {
  let hash = 0

  for (let index = 0; index < value.length; index += 1) {
    hash = (hash * 31 + value.charCodeAt(index)) >>> 0
  }

  return hash
}

// Returns an hsl() string for a named profile, or null for default/empty
// (rendered neutral / untagged).
export function profileColor(name: null | string | undefined): null | string {
  const key = (name ?? '').trim()

  if (!key || key === 'default') {
    return null
  }

  const hue = hashString(key) % 360

  return `hsl(${hue} ${PROFILE_TAG_SATURATION}% ${PROFILE_TAG_LIGHTNESS}%)`
}

// A profile's effective color: a user-picked override wins, else the
// deterministic hue. Default/empty stays neutral (null) regardless.
export function resolveProfileColor(
  name: null | string | undefined,
  overrides: Record<string, string>
): null | string {
  const key = (name ?? '').trim()

  if (!key || key === 'default') {
    return null
  }

  return overrides[key] ?? profileColor(key)
}

// Curated swatches for the rail color picker — evenly spaced hues at the same
// saturation/lightness as the deterministic palette, so picks stay cohesive.
export const PROFILE_SWATCHES: readonly string[] = Array.from(
  { length: 12 },
  (_, index) => `hsl(${index * 30} ${PROFILE_TAG_SATURATION}% ${PROFILE_TAG_LIGHTNESS}%)`
)

// Translucent fill derived from a profile color, for tag backgrounds.
export function profileColorSoft(color: string, percent = 16): string {
  return `color-mix(in srgb, ${color} ${percent}%, transparent)`
}
