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

// Translucent fill derived from a profile color, for tag backgrounds.
export function profileColorSoft(color: string, percent = 16): string {
  return `color-mix(in srgb, ${color} ${percent}%, transparent)`
}
