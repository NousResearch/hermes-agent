import type { ChatMessage } from '@/lib/chat-messages'
import { $gateway } from '@/store/gateway'
import { notifyError } from '@/store/notifications'
import { requestForOwnedSession } from '@/store/session-states'
import { $activeSessionId, $messages, setMessages } from '@/store/session'
import type { MessageReaction } from '@/types/hermes'

/** The six iOS Tapback defaults, in Apple's order. */
export const QUICK_REACTIONS = ['❤️', '👍', '👎', '😂', '‼️', '❓'] as const

interface MessageReactResponse {
  row_id: number
  reactions: MessageReaction[]
}

/** Apply the local half of a tapback: one reaction per author, re-tap retracts. */
export function applyReaction(
  reactions: MessageReaction[] | undefined,
  emoji: null | string,
  author: MessageReaction['author']
): MessageReaction[] {
  const current = reactions ?? []
  const previous = current.find(reaction => reaction.author === author)
  const without = current.filter(reaction => reaction.author !== author)

  if (!emoji || previous?.emoji === emoji) {
    return without
  }

  return [...without, { emoji, author, at: Date.now() / 1000 }]
}

function writeReactions(messageId: string, reactions: MessageReaction[], rowId?: number) {
  // A NEW ChatMessage object per change is load-bearing: the runtime
  // repository caches normalized ThreadMessages in a WeakMap keyed by
  // ChatMessage identity, so a mutation in place renders stale.
  // Keyed by the renderer id, not rowId: a live message has no rowId yet.
  setMessages(messages =>
    messages.map(message =>
      message.id === messageId ? { ...message, reactions, ...(rowId === undefined ? {} : { rowId }) } : message
    )
  )
}

/**
 * Toggle *author*'s reaction on a persisted message.
 *
 * Optimistic: paints immediately, then lets the backend's returned list win.
 * A failed write rolls back to the snapshot (desktop AGENTS.md — "be optimistic,
 * then honest").
 */
export async function toggleMessageReaction(
  message: ChatMessage,
  emoji: null | string,
  author: MessageReaction['author'] = 'user'
): Promise<void> {
  // A live message hasn't round-tripped through a resume yet, so it carries no
  // rowId. Rather than disable the affordance (which made reactions invisible
  // in any active conversation), let the backend resolve the newest row of
  // this role — which is exactly the message being reacted to.
  const rowId = message.rowId
  const sessionId = $activeSessionId.get()
  const gateway = $gateway.get()

  if (!sessionId || !gateway) {
    notifyError(new Error(!sessionId ? 'No active session' : 'Gateway not connected'), 'Could not react')

    return
  }

  // Bound (not wrapped) so the ambient fallback keeps the exact call shape
  // gateway.request callers assert on — mirrors approval.respond routing.
  const ambientRequest = gateway.request.bind(gateway)

  const snapshot = $messages.get().find(m => m.id === message.id)?.reactions

  writeReactions(message.id, applyReaction(snapshot, emoji, author))

  try {
    // Route through the session's OWNER, not the ambient active gateway: the
    // window may be foregrounding a different profile/connection than the one
    // that owns this session (secondary-profile chats, Bot-Mode tiles,
    // post-reconnect rehydration). Dispatching on the ambient socket made the
    // backend that never held the runtime answer 4040 "message not found"
    // even though the row exists in the owning profile's state DB (#80670).
    // requestForOwnedSession resolves the exact owner route and fails closed;
    // the ambient request stays the fallback for legacy single-profile
    // setups where the owner cannot be named.
    const result = await requestForOwnedSession<MessageReactResponse>(
      sessionId,
      ambientRequest,
      'message.react',
      {
        session_id: sessionId,
        ...(rowId === undefined ? { newest_role: message.role } : { row_id: rowId }),
        emoji,
        author
      }
    )

    // Learn the row id from the response so later toggles address it directly.
    writeReactions(message.id, result?.reactions ?? [], result?.row_id)
  } catch (err) {
    // Be optimistic, THEN honest: a rejected write rolls back visibly and says
    // why, instead of the reaction quietly vanishing (desktop AGENTS.md).
    writeReactions(message.id, snapshot ?? [])
    notifyError(err, 'Could not react')
  }
}
