import { textWithoutReferenceLines } from '@/components/assistant-ui/reference-kinds'
import { type ChatMessage, chatMessageText, sameAttachmentTurn } from '@/lib/chat-messages'

import { conflictingTranscriptIdentity } from './pending-turn-identity'

// Only typed steering rows may unwrap the model's delivery envelope. Ordinary
// prompts containing a lookalike marker remain literal text.
const steeringText = (message: ChatMessage): string => {
  const text = textWithoutReferenceLines(chatMessageText(message))

  return message.steering
    ? (
        text.match(/^\[OUT-OF-BAND USER MESSAGE[^\]]*\]\s*([\s\S]*?)\s*\[\/OUT-OF-BAND USER MESSAGE\]$/)?.[1] ?? text
      ).trim()
    : text
}

/** Pair occurrences after the acknowledged boundary, consuming each receipt once. */
export function acknowledgedSteeringMessages(candidates: ChatMessage[], local: ChatMessage[]): Set<string> {
  const acknowledged = new Set<string>()
  let cursor = 0

  for (const message of local) {
    if (message.role !== 'user') {
      continue
    }

    const index = candidates.findIndex(
      (candidate, at) =>
        at >= cursor &&
        !conflictingTranscriptIdentity(message, candidate) &&
        (!candidate.steering || message.steering === true) &&
        (steeringText(candidate) === steeringText(message) || sameAttachmentTurn(candidate, message))
    )

    if (index < 0) {
      continue
    }

    cursor = index + 1

    if (message.steering) {
      acknowledged.add(message.id)
    }
  }

  return acknowledged
}

/** True when the server has already acknowledged this local optimistic user row.
 *  A typed steering row counts once its receipt was paired (by id); an ordinary
 *  prompt keeps the tolerant text/attachment candidate match. */
export function acknowledgedByServer(
  message: ChatMessage,
  acknowledgedSteers: ReadonlySet<string>,
  acknowledgedUserCandidates: ChatMessage[]
): boolean {
  if (message.steering) {
    return acknowledgedSteers.has(message.id)
  }

  return acknowledgedUserCandidates.some(
    candidate =>
      // #122079: the tolerant arm widens the TEXT compare only — it stays
      // inside the identity gate, so a rowId-bearing optimistic row is
      // never swallowed by a committed row it provably is not (a genuine
      // repeat of the same captioned paste). The rowId-less paste from
      // #120978 carries no identity and keeps matching tolerantly.
      !conflictingTranscriptIdentity(message, candidate) &&
      (textWithoutReferenceLines(chatMessageText(candidate)) ===
        textWithoutReferenceLines(chatMessageText(message)) ||
        sameAttachmentTurn(candidate, message))
  )
}

/** True when a non-steering local row's text is already carried verbatim by its
 *  authoritative twin — the PR's steering rows are exempt (their durable twin
 *  arrives through a different acknowledgement path). */
export function authoritativeTwinCarriesText(message: ChatMessage, authoritative: ChatMessage): boolean {
  return (
    !message.steering &&
    textWithoutReferenceLines(chatMessageText(authoritative)) ===
      textWithoutReferenceLines(chatMessageText(message))
  )
}
