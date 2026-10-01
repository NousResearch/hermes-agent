import type { RosterProfileMetadata } from './connection-registry'

/** Public profile presentation from the responding owner, shared by every roster source. */
export function rosterProfileMetadata(profiles: unknown): Record<string, RosterProfileMetadata> | undefined {
  if (!Array.isArray(profiles)) {return undefined}

  return Object.fromEntries(profiles.map(profile => {
    const name = String(profile?.name || '').trim()

    if (!name) {return null}
    const metadata: RosterProfileMetadata = {}

    if (typeof profile?.display_name === 'string' && profile.display_name.trim()) {
      metadata.display_name = profile.display_name.trim()
    }

    if (typeof profile?.title === 'string' && profile.title.trim()) {
      metadata.title = profile.title.trim()
    }

    if (profile?.ui_meta && typeof profile.ui_meta === 'object') {
      metadata.ui_meta = profile.ui_meta
    }

    if (typeof profile?.has_avatar === 'boolean') {
      metadata.has_avatar = profile.has_avatar
    }

    return [name, metadata] as const
  }).filter((entry): entry is readonly [string, RosterProfileMetadata] => Boolean(entry)))
}
