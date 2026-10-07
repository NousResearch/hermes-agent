import { requestComposerGetDraft, requestComposerSetDraft } from '@/app/chat/composer/focus'
import { type ChatMessage, chatMessageText } from '@/lib/chat-messages'
import { $restoredDraftNotice, stashSessionDraft, takeSessionDraft } from '@/store/composer'

/** The submitted user prompt to put back, or null when the composer already
 *  holds text (never clobber what the user started typing). */
export function submittedPromptToRestore(
  messages: ChatMessage[],
  liveDraft: string | null | undefined
): string | null {
  if ((liveDraft ?? '').trim()) {
    return null
  }

  for (let index = messages.length - 1; index >= 0; index--) {
    const message = messages[index]

    if (!message || message.role !== 'user' || message.hidden) {
      continue
    }

    const text = chatMessageText(message).trim()

    if (text) {
      return text
    }
  }

  return null
}

/** An interrupted turn cleared the composer on send. Put that prompt back
 *  into the empty composer for these session ids and publish the undoable
 *  notice. No-op when any addressed draft already has text or attachments. */
export async function restoreInterruptedSubmittedPrompt(
  sessionIds: string[],
  messages: ChatMessage[]
): Promise<boolean> {
  const ids = [...new Set(sessionIds.map(id => id.trim()).filter(Boolean))]

  if (ids.length === 0) {
    return false
  }

  const live = await requestComposerGetDraft(ids)
  const prompt = submittedPromptToRestore(messages, live?.text)

  if (!prompt) {
    return false
  }

  for (const id of ids) {
    const stashed = takeSessionDraft(id)

    if (stashed.text.trim() || stashed.attachments.length > 0) {
      return false
    }
  }

  for (const id of ids) {
    stashSessionDraft(id, prompt, [])
  }

  $restoredDraftNotice.set({ fromKey: ids[0]!, kind: 'interrupt', sessionKeys: ids, text: prompt })
  await requestComposerSetDraft(ids, prompt)

  return true
}
