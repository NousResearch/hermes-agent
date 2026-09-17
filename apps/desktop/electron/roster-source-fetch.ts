import type { RosterProfileMetadata } from './connection-registry'

export function rosterProfileMetadata(profile: unknown): RosterProfileMetadata {
  if (!profile || typeof profile !== 'object') {
    return {}
  }

  const row = profile as Record<string, unknown>
  const metadata: RosterProfileMetadata = {}
  const displayName = typeof row.display_name === 'string' ? row.display_name.trim() : ''
  const botTitle = typeof row.bot_title === 'string' ? row.bot_title.trim() : ''
  const legacyTitle = typeof row.title === 'string' ? row.title.trim() : ''

  if (displayName) {
    metadata.display_name = displayName
  }

  if (botTitle || legacyTitle) {
    metadata.title = botTitle || legacyTitle
  }

  if (row.ui_meta && typeof row.ui_meta === 'object' && !Array.isArray(row.ui_meta)) {
    metadata.ui_meta = row.ui_meta as Record<string, unknown>
  }

  if (typeof row.has_avatar === 'boolean') {
    metadata.has_avatar = row.has_avatar
  }

  return metadata
}

export async function fetchRosterSourceData<T>(
  fetchProfiles: () => Promise<T>,
  fetchInstallId: () => Promise<string | undefined>
): Promise<{ body: T; installId: string | undefined }> {
  const [body, installId] = await Promise.all([fetchProfiles(), fetchInstallId()])

  return { body, installId }
}
