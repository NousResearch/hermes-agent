import { stashSessionDraft, takeSessionDraft } from '@/store/composer'

/**
 * A dictation that finished transcribing after its composer was gone — the user
 * switched chat/bot mid-transcription, so the hook's owner ref was cleared and
 * the transcript had nowhere to go. Keep the text in that session's persisted
 * draft instead of dropping it, so it is waiting when the user comes back.
 *
 * Addressed by the session the dictation started in, never by "whatever is
 * active now": the point is that this text belongs to the other chat.
 */
export function salvageDictation(sessionId: string | null | undefined, text: string): boolean {
  const incoming = text.trim()

  if (!incoming) {
    return false
  }

  const existing = takeSessionDraft(sessionId)
  const current = existing.text.trim()

  stashSessionDraft(sessionId, current ? `${current} ${incoming}` : incoming, existing.attachments)

  return true
}
