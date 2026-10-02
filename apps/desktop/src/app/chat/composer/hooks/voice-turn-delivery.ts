import type { ComposerAttachment } from '@/store/composer'

export type VoiceTurnDelivery = 'drafted' | 'queued' | 'submitted'

interface DeliverVoiceTurnArgs {
  busy: boolean
  enqueue: (key: string, payload: { attachments: ComposerAttachment[]; text: string }) => unknown
  insertText: (text: string) => void
  onSubmit: (text: string) => Promise<boolean> | boolean
  /** Called with the queue entry id when the turn was queued. */
  onQueued?: (id: string) => void
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
  onQueued,
  onSubmit,
  queueKey,
  text
}: DeliverVoiceTurnArgs): Promise<VoiceTurnDelivery> {
  const keep = (): VoiceTurnDelivery => {
    const entry = queueKey ? enqueue(queueKey, { attachments: [], text }) : null

    if (entry) {
      const id = (entry as { id?: unknown }).id

      if (typeof id === 'string') {
        onQueued?.(id)
      }

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

interface ReclaimVoiceAsidesArgs {
  getQueued: (key: string) => { id: string; text: string }[]
  ids: string[]
  insertText: (text: string) => void
  key: string
  remove: (key: string, id: string) => boolean
}

/**
 * On a spoken "stop": pull this conversation's voice-queued asides out of the
 * queue and into the composer. A stop parks the queue, so those entries would
 * otherwise sit ahead of whatever the user says next (or, unparked, auto-send
 * and restart the agent). In the composer they are kept but not sent.
 */
export function reclaimVoiceAsides({ getQueued, ids, insertText, key, remove }: ReclaimVoiceAsidesArgs): void {
  const queued = getQueued(key)
  const texts: string[] = []

  for (const id of ids) {
    const entry = queued.find(e => e.id === id)

    if (entry && remove(key, id)) {
      texts.push(entry.text)
    }
  }

  if (texts.length) {
    insertText(texts.join(' '))
  }
}
