import type { SessionInfo } from '@/types/hermes'

/** Stored ids and lineage ids are unique only within their gateway/profile. */
export function sessionRowKey(row: Pick<SessionInfo, 'id' | 'profile' | 'connection_id'>, id = row.id): string {
  return JSON.stringify([row.connection_id || '', row.profile?.trim() || 'default', id])
}
