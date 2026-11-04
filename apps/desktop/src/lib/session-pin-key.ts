import type { SessionInfo } from '@/types/hermes'

/** Foreign pin intents outlive the active gateway's id-only mirror. Include
 *  their owner so a copied session id on another gateway/profile stays alone. */
export function sessionPinWriteKey(row: SessionInfo): string {
  return JSON.stringify([
    row.connection_id?.trim() || '',
    row.profile?.trim() || 'default',
    row._lineage_root_id || row.id
  ])
}
