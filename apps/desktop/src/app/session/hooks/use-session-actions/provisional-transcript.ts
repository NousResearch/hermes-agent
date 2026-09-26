import type { SessionInfo } from '@/hermes'
import type { ChatMessage } from '@/lib/chat-messages'
import { $messages, setMessages } from '@/store/session'
import type { SessionProfileRoute } from '@/store/session-request-router'
import { dropTranscriptTail, loadTranscriptTail, type TranscriptTailScope } from '@/store/transcript-tail-cache'

export function transcriptRestScope(
  owner: SessionProfileRoute | undefined,
  stored: SessionInfo | undefined,
  ambientConnectionId: string
): TranscriptTailScope | undefined {
  const connectionId = owner ? owner.connectionId : stored?.connection_id || ambientConnectionId
  const profile = owner?.targetProfile || owner?.profile || stored?.profile

  return connectionId ? { connectionId, profile: profile || 'default' } : profile || undefined
}

/** Display-only cache paint: never becomes runtime or persisted-display authority. */
export function provisionalTranscriptPaint(storedSessionId: string, isCurrent: () => boolean) {
  let messages: ChatMessage[] | null = null
  let scopeKey: string | undefined

  return {
    get messages() {
      return messages
    },
    paint(scope: TranscriptTailScope | undefined) {
      if (!isCurrent()) {
        return
      }

      const nextKey = JSON.stringify(scope)

      if (messages !== null && nextKey !== scopeKey) {
        if ($messages.get() === messages) {
          setMessages([])
        }

        messages = null
      }

      scopeKey = nextKey

      // An unknown owner is not permission to read a legacy unscoped entry.
      if (!scope || messages !== null || $messages.get().length > 0) {
        return
      }

      messages = loadTranscriptTail(storedSessionId, scope)

      if (messages) {
        setMessages(messages)
      }
    },
    /** Roll back an unreconciled paint (#120215): when every authoritative
     *  source failed (resume RPC + REST fallback) and the untouched
     *  provisional paint is still all that's showing, it is unproven stale —
     *  not transcript. Clear it and evict the entry so the retry / next wake
     *  re-fetches instead of re-painting the same frozen tail forever
     *  (detached websocket). A no-op when nothing painted, or when an
     *  authoritative transcript already replaced the paint (reference check:
     *  the success path overwrote the entry with fresh truth — keep it). */
    rollback(scope: TranscriptTailScope | undefined) {
      if (messages === null) {
        return
      }

      const painted = messages
      messages = null

      if ($messages.get() === painted) {
        setMessages([])
        dropTranscriptTail(storedSessionId, scope)
      }
    }
  }
}
