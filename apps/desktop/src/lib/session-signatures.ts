/**
 * Cheap signature compares for poll loops — swap the atom (and re-render)
 * only when the rows/transcript actually changed.
 */

import type { SessionInfo, SessionMessage } from '@/hermes'

export function sameCronSignature(a: SessionInfo[], b: SessionInfo[]): boolean {
  if (a.length !== b.length) {
    return false
  }

  return a.every((session, i) => {
    const other = b[i]

    return (
      other != null &&
      session.id === other.id &&
      session._lineage_root_id === other._lineage_root_id &&
      session.title === other.title &&
      session.source === other.source &&
      session.profile === other.profile &&
      session.preview === other.preview &&
      session.message_count === other.message_count &&
      session.last_active === other.last_active &&
      session.ended_at === other.ended_at
    )
  })
}

// FNV-1a over role/timestamp/content.
function hashString(hash: number, value: string): number {
  let next = hash

  for (let i = 0; i < value.length; i++) {
    next ^= value.charCodeAt(i)
    next = Math.imul(next, 16777619)
  }

  return next >>> 0
}

/** Transcript fingerprint for the active-messaging-session poll. */
export function sessionMessagesSignature(messages: SessionMessage[]): string {
  let hash = 2166136261

  for (const m of messages) {
    hash = hashString(hash, m.role)
    hash = hashString(hash, String(m.timestamp ?? ''))
    hash = hashString(hash, typeof m.content === 'string' ? m.content : (JSON.stringify(m.content) ?? ''))
  }

  return `${messages.length}:${hash}`
}
