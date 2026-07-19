import { getSessionMessages } from '@/hermes'
import { type ChatMessage, preserveLocalAssistantErrors, toChatMessages } from '@/lib/chat-messages'
import { $sessions, sessionMatchesStoredId } from '@/store/session'

/**
 * Multi-window stale guard (#65047): another Desktop window (or tile) may own
 * a fresher transcript while this view still shows an open-time snapshot.
 *
 * Returns a refreshed ChatMessage[] when the authoritative transcript is ahead
 * of the local view; otherwise null. Fail-open on read errors so a wrong-backend
 * or missing-profile route never soft-locks the composer.
 *
 * Always resolves the session's owning profile from `$sessions` so cross-profile
 * GET routing in {@link getSessionMessages} hits the right backend/state.db.
 */
export async function refreshIfTranscriptStale(
  storedSessionId: string,
  localMessages: ChatMessage[],
  options?: { excludeMessageId?: string }
): Promise<ChatMessage[] | null> {
  const baseline = options?.excludeMessageId
    ? localMessages.filter(message => message.id !== options.excludeMessageId)
    : localMessages
  const localCount = baseline.length
  const storedProfile = $sessions.get().find(session => sessionMatchesStoredId(session, storedSessionId))?.profile

  try {
    const remote = await getSessionMessages(storedSessionId, storedProfile)

    // Compare ChatMessage lengths (tools are folded by toChatMessages), not raw
    // SessionMessage counts — otherwise a tool-using transcript looks "ahead"
    // forever and soft-locks every subsequent send.
    const remoteChat = toChatMessages(remote.messages)

    if (remoteChat.length > localCount) {
      return preserveLocalAssistantErrors(remoteChat, baseline)
    }
  } catch {
    // Authoritative check failed: do not soft-lock the composer.
  }

  return null
}
