import { atom, computed, type ReadableAtom } from 'nanostores'

import { isOnboardingChat, onboardingChatStore } from './onboarding-session'

/** Answers a card parked on `sessionId` with composer text: null when no card of this kind takes it,
 *  else a promise of whether the answer reached the backend. */
export type ParkedCardAnswer = (sessionId: string, text: string) => null | Promise<boolean>

/** One card kind: its typed-answer handler, whether a card of it waits on a session, and the store that
 *  predicate reads (watched so the composer redraws when a card parks or settles). */
export interface ParkedCardKind {
  $requests: ReadableAtom<object>
  answer: ParkedCardAnswer
  /** Takes typed text only in the onboarding chat; elsewhere the text is sent as a normal message. */
  onboardingOnly?: boolean
  /** Closes the card unanswered, so a message that cannot answer it (attachments, a slash command) runs. */
  skip?: (sessionId: string) => boolean
  /** Closes the card unanswered because the user sent a message instead. The card's turn ends on that
   *  answer, so the message runs as the next turn. True when a card closed. */
  yieldToMessage?: (sessionId: string) => boolean
  waiting: (sessionId: string) => boolean
}

const kinds = new Map<string, ParkedCardKind>()

const kindsFor = (sessionId: string) => {
  const onboarding = isOnboardingChat(sessionId)

  return [...kinds.values()].filter(kind => onboarding || !kind.onboardingOnly)
}

// Bumped when a kind registers, unregisters or its store changes, so per-session views re-ask every kind.
const $revision = atom(0)
const bump = () => $revision.set($revision.get() + 1)

/** Register a card kind under `id`. Returns the unregister function. */
export function registerParkedCardAnswer(id: string, kind: ParkedCardKind): () => void {
  kinds.set(id, kind)
  const unlisten = kind.$requests.listen(bump)
  bump()

  return () => {
    unlisten()

    if (kinds.get(id) === kind) {
      kinds.delete(id)
      bump()
    }
  }
}

/**
 * Offer composer text to every card waiting on `sessionId`, in registration order. True when one
 * took it as its answer: the composer then neither sends nor steers the text, and `onFailed` runs
 * if that answer never reaches the backend, so the composer can give the words back.
 */
export function answerParkedCard(sessionId: string, text: string, onFailed: () => void = () => {}): boolean {
  for (const kind of kindsFor(sessionId)) {
    const sent = kind.answer(sessionId, text)

    if (sent) {
      void sent
        .then(delivered => {
          if (!delivered) {
            onFailed()
          }
        })
        .catch(onFailed)

      return true
    }
  }

  return false
}

/** Close every card waiting on `sessionId` that gives its turn to a message sent instead of an answer. True
 *  when one closed: the message then waits for that turn to end and runs as the next one, never a steer. */
export function yieldParkedCardsToMessage(sessionId: string): boolean {
  let yielded = false

  for (const kind of kindsFor(sessionId)) {
    if (kind.waiting(sessionId) && kind.yieldToMessage?.(sessionId)) {
      yielded = true
    }
  }

  return yielded
}

/** Close every card waiting on `sessionId` that a message cannot answer (attachments, a slash command). */
export function skipParkedCards(sessionId: string): void {
  for (const kind of kindsFor(sessionId)) {
    if (kind.waiting(sessionId)) {
      kind.skip?.(sessionId)
    }
  }
}

/** Whether any card waits on `sessionId`, so typed text would answer it. */
export function parkedCardWaiting(sessionId: string): boolean {
  return kindsFor(sessionId).some(kind => kind.waiting(sessionId))
}

/** Whether a card waits in the onboarding chat, as a store: the composer offers Send for typed
 *  answers only there; every other chat keeps its normal controls. */
export const sessionOnboardingCardWaiting = (sessionId: null | string) =>
  computed([$revision, onboardingChatStore(sessionId)], (_, onChat) => onChat && parkedCardWaiting(sessionId ?? ''))
