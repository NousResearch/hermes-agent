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
      session.ended_at === other.ended_at &&
      // A workspace move need not change activity or message counts. Let its
      // authoritative row reach the project overlays and move-target menu.
      session.cwd === other.cwd &&
      session.git_repo_root === other.git_repo_root &&
      session.git_branch === other.git_branch &&
      // Row STATE, not just row content: session-pin-sync reconciles the
      // sidebar's pins against `pinned` on the rows in this atom, so a page
      // whose only delta is a flag has to swap in or the reconciler reads a
      // frozen copy forever. An idle conversation never moves any of the
      // fields above again, which is exactly when a pin gets toggled (#76919).
      session.pinned === other.pinned &&
      session.archived === other.archived
    )
  })
}

// FNV-1a over the row fields below.
function hashString(hash: number, value: string): number {
  let next = hash

  for (let i = 0; i < value.length; i++) {
    next ^= value.charCodeAt(i)
    next = Math.imul(next, 16777619)
  }

  return next >>> 0
}

/** Cheap sidebar-row fingerprint. Used to skip a 120-row transcript fetch
 *  when message_count, last_active, and preview have not moved (#95767). */
export function sessionListFingerprint(session: {
  last_active?: number
  message_count?: number
  preview?: null | string
}): string {
  return `${session.message_count ?? 0}:${session.last_active ?? 0}:${session.preview ?? ''}`
}

/** Row fields the transcript hydration (`toChatMessages`) reads beyond
 *  role+timestamp+raw content. "Unchanged signature" must imply unchanged
 *  rendered projection, so an authoritative presentation-only correction —
 *  corrected public text, a hidden row, a reaction — still repaints (#140396). */
const MESSAGE_SIGNATURE_FIELDS: readonly (keyof SessionMessage)[] = [
  'args',
  'codex_message_items',
  'codex_reasoning_items',
  'context',
  'display_commentary',
  'display_content',
  'display_kind',
  'display_metadata',
  'display_reasoning',
  'labels',
  'name',
  'reasoning',
  'reasoning_content',
  'reasoning_details',
  'row_id',
  'text',
  'tool_call_id',
  'tool_call_labels',
  'tool_name',
  'tool_calls'
]

// An absent field and an explicit `null` hydrate differently
// (`display_content !== undefined ? display_content : fallback`), so the
// sentinel keeps them apart; JSON quoting keeps `"1"` and `1` apart.
const ABSENT_FIELD = '\u0000'

function fieldText(value: unknown): string {
  return JSON.stringify(value) ?? ABSENT_FIELD
}

/** Transcript fingerprint for the active-messaging-session poll. */
export function sessionMessagesSignature(messages: SessionMessage[]): string {
  let hash = 2166136261

  for (const m of messages) {
    hash = hashString(hash, m.role)
    hash = hashString(hash, String(m.timestamp ?? ''))
    hash = hashString(hash, fieldText(m.content))

    for (const field of MESSAGE_SIGNATURE_FIELDS) {
      hash = hashString(hash, fieldText(m[field]))
    }
  }

  return `${messages.length}:${hash}`
}
