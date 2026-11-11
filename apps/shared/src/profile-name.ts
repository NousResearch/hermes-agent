/** Presentation metadata only; `name` remains the routing and storage identity. */
export interface ProfileNameMetadata {
  name: string
  bot_title?: string | null
  display_name?: string | null
  ui_meta?: unknown
}

/** Bot title → core display name → legacy identity. Never use this label for routing. */
export function profileVisibleName(profile: ProfileNameMetadata): string {
  const meta = profile.ui_meta
  const bots = meta && typeof meta === 'object' && 'hermes-bots' in meta ? meta['hermes-bots'] : undefined
  const title = bots && typeof bots === 'object' && 'title' in bots ? bots.title : undefined
  const botTitle = typeof profile.bot_title === 'string' ? profile.bot_title.trim() : ''

  return botTitle || (typeof title === 'string' ? title.trim() : '') ||
    profile.display_name?.trim() || profile.name
}
