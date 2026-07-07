import type { SessionInfo } from '@/hermes'

/** Whether an in-flight profile-restore probe has been superseded and must not
 *  navigate. True when a newer switch bumped the token, or when the SELECTION
 *  CHANGED while probing (the user actively opened a chat / started a resume).
 *  The pre-switch profile's still-open session legitimately sits in
 *  $selectedStoredSessionId when the switch fires, so an unchanged leftover
 *  selection must NOT abort the restore — only a change does. */
export function profileRestoreSuperseded(args: {
  currentToken: number
  requestToken: number
  selectedAtRequest: null | string
  selectedNow: null | string
}): boolean {
  return args.currentToken !== args.requestToken || args.selectedNow !== args.selectedAtRequest
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
