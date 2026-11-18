import { textWithoutReferenceLines } from '@/components/assistant-ui/reference-kinds'
import { type ChatMessage, chatMessageText } from '@/lib/chat-messages'
import { isLiveTailReplyId } from '@/lib/spoken-reply'

/**
 * A live-turn row — the gateway's text-only `inflight` projection, a
 * still-streaming local bubble, or an interim row sealed inside the running
 * turn — as opposed to a committed transcript row.
 */
export function isLiveTailRow(message: ChatMessage): boolean {
  return message.pending === true || isLiveTailReplyId(message.id) || message.interim === true
}

/**
 * The contiguous RUN of optimistic `user-*` rows that belong to the live turn.
 *
 * A mid-turn redirect inserts its correction as a second optimistic user row
 * directly before the live reply, so one turn can own a contiguous run of them.
 * Preserving only the newest keeps the correction and drops the prompt that
 * started the turn. Widen to the run — but only the contiguous one: any `user-*`
 * row separated by a committed assistant reply is stale post-compression
 * history, which is what the newest-only rule exists to discard. A live-tail
 * assistant row between the prompt and its correction (#73793) is still the same
 * turn's run; only a committed reply ends it.
 */
export function liveOptimisticUserRun(
  previousMessages: ChatMessage[],
  acknowledgedLocalIndex: number
): Set<ChatMessage> {
  const newestOptimisticUser = [...previousMessages]
    .reverse()
    .find(message => message.role === 'user' && message.id.startsWith('user-'))

  const liveOptimisticUsers = new Set<ChatMessage>()

  if (!newestOptimisticUser) {
    return liveOptimisticUsers
  }

  for (let index = previousMessages.indexOf(newestOptimisticUser); index > acknowledgedLocalIndex; index -= 1) {
    const candidate = previousMessages[index]

    if (candidate.role === 'user' && candidate.id.startsWith('user-')) {
      liveOptimisticUsers.add(candidate)

      continue
    }

    if (candidate.role === 'assistant' && isLiveTailRow(candidate)) {
      continue
    }

    break
  }

  return liveOptimisticUsers
}

/**
 * The retained safety-refused prompts in a committed transcript. A refused
 * prompt hydrates as a greyed `system` content_refused row (NOTICE_DISPLAY_KINDS)
 * carrying the user's own text, never a `user` row.
 */
export function refusedPromptCounterparts(messages: ChatMessage[]): ChatMessage[] {
  return messages.filter(message => message.role === 'system' && message.contentRefused)
}

/**
 * Whether an optimistic `user-*` prompt is already represented in the committed
 * window and should be dropped rather than re-appended as unsent input:
 *
 *  • it is not part of the live contiguous optimistic run (stale
 *    post-compression history), or
 *  • its stored counterpart is the greyed content_refused row (same prompt
 *    text). The acknowledged-user match can't see that row — it scans `user`
 *    rows, and a canonical submit can leave the optimistic prompt without a
 *    rowId — so without this the twin is re-appended beside the greyed block on
 *    warm resume.
 */
export function optimisticUserAlreadyRepresented(
  message: ChatMessage,
  liveOptimisticUsers: Set<ChatMessage>,
  refusedCounterparts: ChatMessage[]
): boolean {
  if (!liveOptimisticUsers.has(message)) {
    return true
  }

  return refusedCounterparts.some(
    refused =>
      textWithoutReferenceLines(chatMessageText(refused)) === textWithoutReferenceLines(chatMessageText(message))
  )
}
