import type { ChatMessage } from '@/lib/chat-messages'
import { $gateway } from '@/store/gateway'
import { notifyError } from '@/store/notifications'
import { $activeSessionId, $messages, $selectedStoredSessionId, setMessages } from '@/store/session'
import { requestForOwnedSession } from '@/store/session-states'
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
  author: MessageReaction['author'] = 'user',
  sessionIdOverride?: null | string
): Promise<void> {
  // A live message hasn't round-tripped through a resume yet, so it carries no
  // rowId. Rather than disable the affordance (which made reactions invisible
  // in any active conversation), let the backend resolve the newest row of
  // this role — which is exactly the message being reacted to.
  const rowId = message.rowId
  // The runtime id is unavailable while a stored conversation is being
  // resumed (and can briefly be cleared during a profile/context switch), but
  // the mounted transcript is still owned by the selected stored session.
  // Use that durable identity as the routing fallback instead of treating a
  // visible conversation as a draft. A genuinely new draft has neither id.
  const sessionId =
    sessionIdOverride === undefined ? ($activeSessionId.get() ?? $selectedStoredSessionId.get()) : sessionIdOverride
  const gateway = $gateway.get()

  if (!sessionId || !gateway) {
    notifyError(new Error(!sessionId ? 'No active session' : 'Gateway not connected'), 'Could not react')

    return
  }

  const snapshot = $messages.get().find(m => m.id === message.id)?.reactions

  writeReactions(message.id, applyReaction(snapshot, emoji, author))

  try {
    // Keep reaction writes on the session's owning backend. The selected
    // stored-id fallback above is especially important here: owner lookup can
    // resolve it even while the runtime binding is still being rebuilt.
    const result = await requestForOwnedSession<MessageReactResponse>(
      sessionId,
      gateway.request.bind(gateway) as typeof gateway.request,
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
