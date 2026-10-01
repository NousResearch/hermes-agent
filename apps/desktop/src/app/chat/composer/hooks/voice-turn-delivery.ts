import type { ComposerAttachment } from '@/store/composer'

export type VoiceTurnDelivery = 'drafted' | 'queued' | 'submitted'

interface DeliverVoiceTurnArgs {
  busy: boolean
  enqueue: (key: string, payload: { attachments: ComposerAttachment[]; text: string }) => unknown
  insertText: (text: string) => void
  onSubmit: (text: string) => Promise<boolean> | boolean
  queueKey: null | string | undefined
  text: string
}

/**
 * Hand a transcribed voice turn to the chat without ever losing it.
 *
 * The submit path refuses (returns false) while the session is busy, and the
 * voice loop used to bail early on `busy` — either way a sentence the user
 * spoke while the agent was working vanished after a successful transcription.
 * Busy → queue it behind the running turn (the composer queue drains on the
 * busy→false edge, same as a typed message sent mid-turn). No queue yet → park
 * it in the composer so the words stay on screen.
 */
export async function deliverVoiceTurn({
  busy,
  enqueue,
  insertText,
  onSubmit,
  queueKey,
  text
}: DeliverVoiceTurnArgs): Promise<VoiceTurnDelivery> {
  const keep = (): VoiceTurnDelivery => {
    if (queueKey && enqueue(queueKey, { attachments: [], text })) {
      return 'queued'
    }

    insertText(text)

    return 'drafted'
  }

  if (busy) {
    return keep()
  }

  return (await onSubmit(text)) === false ? keep() : 'submitted'
}
