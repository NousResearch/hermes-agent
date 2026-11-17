import { textWithoutReferenceLines } from '@/components/assistant-ui/reference-kinds'
import { type ChatMessage, chatMessageText } from '@/lib/chat-messages'

import { isLiveTailRow } from './optimistic-user-dedup'
import { conflictingTranscriptIdentity } from './pending-turn-identity'

const answerText = (message: ChatMessage) => textWithoutReferenceLines(chatMessageText(message)).trim()

/**
 * A committed tool round with no prose (`content = ''`, `tool_calls` set)
 * hydrates as a text-less shell. When no later prose row folds into it, the
 * settled local final bubble is the only copy of the answer the user watched
 * stream, and dropping it for the shell deletes the reply on sync (#123047).
 * Interim narration and structural-only local rows never claim the slot, and
 * a retained error row is never a shell. Kept apart from
 * `localPendingSupersedes`, whose stricter live-only rule also gates resume.
 */
export const settledReplyOverEmptyShell = (local: ChatMessage, authoritative: ChatMessage): boolean =>
  local.role === 'assistant' &&
  local.pending !== true &&
  local.interim !== true &&
  !local.error &&
  authoritative.role === 'assistant' &&
  !isLiveTailRow(authoritative) &&
  !authoritative.error &&
  answerText(authoritative).length === 0 &&
  answerText(local).length > 0

/**
 * A settled reply bubble whose ordinal-matched page row is a text-less
 * reasoning/tool FOLD of this same turn. The fold addresses the turn's FIRST row
 * (`rowId` = that first assistant row) while the reply addresses its final prose
 * row (a completion receipt binds that one as `rowId`), so
 * `conflictingTranscriptIdentity` rejects the pair even though the two are one
 * bubble — and the page row is text-less precisely because the prose row it would
 * fold is the very reply the page does not carry yet.
 *
 * This is the settled-row sibling of `isSettledUnpersistedAssistant` (in
 * `utils.ts`) with the two clauses that excluded it inverted: the row may carry a
 * bound `rowId`, and the ordinal may be occupied — but only by a text-less shell.
 * It admits the reply so `settledReplyOverEmptyShell` can replace the shell;
 * without it the reply is dropped and the shell stays on screen as a
 * reasoning-only turn with no answer and no footer (the "Thought for…" state).
 * Scoped to a text-less authoritative row, so a page carrying a real answer at
 * the ordinal is untouched and a superseded or compacted reply is never
 * resurrected.
 */
export const isSettledReplyOverOrdinalShell = (
  message: ChatMessage,
  ordinalShell: ChatMessage | undefined,
  isLastRow: boolean
): boolean =>
  isLastRow &&
  message.role === 'assistant' &&
  message.pending !== true &&
  message.interim !== true &&
  message.hidden !== true &&
  chatMessageText(message).trim() !== '' &&
  ordinalShell !== undefined &&
  settledReplyOverEmptyShell(message, ordinalShell)

/**
 * The authoritative row a local reply pairs with at its role ordinal, or
 * `undefined` when there is none. A genuine identity conflict (a different turn
 * at the same ordinal) blocks the pair — except for the text-less fold a settled
 * reply is allowed to replace, which carries the turn's first row id rather than
 * the reply's final one.
 */
export const authoritativeOverOrdinal = (
  message: ChatMessage,
  ordinalMatch: ChatMessage | undefined,
  admitsSettledReplyOverShell: boolean
): ChatMessage | undefined =>
  ordinalMatch && (!conflictingTranscriptIdentity(message, ordinalMatch) || admitsSettledReplyOverShell)
    ? ordinalMatch
    : undefined
