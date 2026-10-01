import type { SessionInfo } from '@/types/hermes'

/**
 * The id safe to remember/restore from a RESOLVED session row: a delegate
 * child (`source === 'subagent'`) is replaced by its parent, everything else
 * keeps its own id. Delegate children are deliberately omitted from the
 * sidebar list (`_LISTABLE_CHILD_SQL`), so remembering one leaves the next
 * cold start split between the highlighted parent and the invisible child the
 * chat area shows (#56983). `/branch` children also carry
 * `parent_session_id` but ARE user-facing — `source`, not parenthood, is the
 * discriminator.
 *
 * `null` means "do not remember" (an orphaned delegate child whose parent is
 * gone).
 */
export function repairRememberedSession(session: SessionInfo): string | null {
  return session.source === 'subagent' ? (session.parent_session_id ?? null) : session.id
}
