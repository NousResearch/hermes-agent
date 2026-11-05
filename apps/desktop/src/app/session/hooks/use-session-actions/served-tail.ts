import { type ChatMessage, stripPendingClarifyProjectionForCache } from '@/lib/chat-messages'
import { saveTranscriptTail, type TranscriptTailScope } from '@/store/transcript-tail-cache'

/** Cache a served transcript as the session's first-paint tail.
 *
 * Only backend truth is cached: the pending/running bit and any synthetic
 * clarify row are a live resume projection and must not outlive the server-side
 * request. `served` is false when the REST page carried no rows — then there is
 * nothing to cache but the painted cache reconciled against itself, and
 * re-saving that is how one foreign page becomes permanent. */
export function saveServedTranscriptTail({
  sessionId,
  messages,
  clarifyRequestId,
  restScope,
  served
}: {
  sessionId: string
  messages: ChatMessage[]
  clarifyRequestId: string | undefined
  restScope: TranscriptTailScope | undefined
  served: boolean
}): void {
  if (!served) return
  saveTranscriptTail(
    sessionId,
    stripPendingClarifyProjectionForCache(messages, clarifyRequestId),
    restScope
  )
}
