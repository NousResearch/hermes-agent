import type { SessionInfo } from '@/hermes'

/**
 * Engine-most-recent fallback for a cold-boot restore whose pinned session id
 * turned out to be dead (deleted/rotated while the app was closed). Prefers a
 * currently-active session (a runtime is live on it right now), then the
 * newest by last_active; skips archived rows and the dead session itself —
 * matched by stored id or lineage root, mirroring sessionMatchesStoredId().
 * Returns null when nothing qualifies (caller falls back to the new-chat
 * route).
 */
export function pickMostRecentSessionId(sessions: SessionInfo[], excludeStoredId: null | string): null | string {
  let best: null | SessionInfo = null

  for (const session of sessions) {
    if (session.archived) {
      continue
    }

    if (excludeStoredId && (session.id === excludeStoredId || session._lineage_root_id === excludeStoredId)) {
      continue
    }

    if (
      !best ||
      (session.is_active && !best.is_active) ||
      (session.is_active === best.is_active && (session.last_active ?? 0) > (best.last_active ?? 0))
    ) {
      best = session
    }
  }

  return best?.id ?? null
}

// Cheap signature compare so a poll only swaps the atom (and re-renders the
// sidebar) when the visible rows actually changed.
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
