import { sessionMatchesStoredId } from '@/store/session'
import type { SessionInfo } from '@/types/hermes'

interface ActiveTranscriptState {
  activeRuntimeId: null | string
  contextSwitching: boolean
  messagesEmpty: boolean
  transcriptStoredSessionId: null | string
}

/**
 * Whether the route points at a different conversation than the selected view.
 *
 * Auto-compression rotates a conversation from its root id to a continuation
 * tip while the durable URL intentionally stays on the root. Those ids are
 * different strings but still the same conversation, so a loaded lineage row
 * must win over the raw comparison. An unknown route remains a mismatch: that
 * is the real navigation case where the old transcript/composer must hide
 * until resume finishes.
 */
export function isRouteSessionMismatch(
  routedSessionId: null | string,
  selectedSessionId: null | string,
  sessions: readonly Pick<SessionInfo, '_lineage_root_id' | 'id'>[],
  activeTranscript?: ActiveTranscriptState
): boolean {
  if (!routedSessionId) {
    return false
  }

  const matchesRoute = (storedSessionId: null | string) =>
    storedSessionId === routedSessionId ||
    Boolean(
      storedSessionId &&
        sessions.some(
          session =>
            sessionMatchesStoredId(session, routedSessionId) && sessionMatchesStoredId(session, storedSessionId)
        )
    )

  // The selected view already owns the routed conversation: a profile or
  // connection switch must not blank it to the splash.
  if (matchesRoute(selectedSessionId)) {
    return false
  }

  // Only the transcript-retention fallback below must be denied while a
  // context switch is in flight; the prior context must not be retained.
  if (activeTranscript?.contextSwitching) {
    return true
  }

  return !(
    activeTranscript?.activeRuntimeId &&
    !activeTranscript.messagesEmpty &&
    matchesRoute(activeTranscript.transcriptStoredSessionId)
  )
}
