/** Answers a card parked on `sessionId` with composer text; true when the text answered it. */
export type ParkedCardAnswer = (sessionId: string, text: string) => boolean

const answers = new Map<string, ParkedCardAnswer>()

/** Register a card kind's typed-answer handler under `id`. Returns the unregister function. */
export function registerParkedCardAnswer(id: string, answer: ParkedCardAnswer): () => void {
  answers.set(id, answer)

  return () => {
    if (answers.get(id) === answer) {
      answers.delete(id)
    }
  }
}

/**
 * Offer composer text to every card waiting on `sessionId`, in registration order. True when one
 * took it as its answer: the composer then neither sends nor steers the text.
 */
export function answerParkedCard(sessionId: string, text: string): boolean {
  for (const answer of answers.values()) {
    if (answer(sessionId, text)) {
      return true
    }
  }

  return false
}
