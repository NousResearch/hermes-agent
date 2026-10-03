import { chatMessageText, renderMediaTags } from '@/lib/chat-messages'

import type { ClientSessionState } from '../../../types'

/**
 * A sealed stream can lose a few characters while the authoritative final
 * remains the same reply. Limit the tolerated edit distance so a separate
 * assistant segment cannot replace a merely similar interim.
 */
export function hasHighTextOverlap(left: string, right: string): boolean {
  const maxLength = Math.max(left.length, right.length)

  if (maxLength < 160) {
    return false
  }

  const maxEdits = Math.max(1, Math.min(32, Math.floor(maxLength * 0.02)))

  if (Math.abs(left.length - right.length) > maxEdits) {
    return false
  }

  const [shorter, longer] = left.length < right.length ? [left, right] : [right, left]

  let previous = Array.from({ length: shorter.length + 1 }, (_, index) =>
    index <= maxEdits ? index : Number.POSITIVE_INFINITY
  )

  let current = new Array<number>(shorter.length + 1).fill(Number.POSITIVE_INFINITY)

  for (let longerIndex = 1; longerIndex <= longer.length; longerIndex += 1) {
    const start = Math.max(1, longerIndex - maxEdits)
    const end = Math.min(shorter.length, longerIndex + maxEdits)
    current.fill(Number.POSITIVE_INFINITY, start, end + 1)
    current[start - 1] = start === 1 ? longerIndex : Number.POSITIVE_INFINITY

    let rowMinimum = Number.POSITIVE_INFINITY

    for (let shorterIndex = start; shorterIndex <= end; shorterIndex += 1) {
      current[shorterIndex] = Math.min(
        previous[shorterIndex] + 1,
        current[shorterIndex - 1] + 1,
        previous[shorterIndex - 1] + Number(longer[longerIndex - 1] !== shorter[shorterIndex - 1])
      )
      rowMinimum = Math.min(rowMinimum, current[shorterIndex])
    }

    if (rowMinimum > maxEdits) {
      return false
    }

    const nextPrevious = current
    current = previous
    previous = nextPrevious
  }

  return previous[shorter.length] <= maxEdits
}

const continuesText = (final: string, existing: string) =>
  final === existing || final.startsWith(existing) || existing.startsWith(final) || hasHighTextOverlap(final, existing)

/**
 * Index of the earlier-turn row a late terminal frame belongs to, or null
 * when the frame is the live turn's own (#101321).
 *
 * A provider stream drop (Grok) can deliver turn A's complete/error after
 * turn B has started streaming. Such a frame shares nothing with B's live
 * bubble but continues a non-interim assistant row above the newest user
 * row. Only a live bubble proves B already has its own reply: without one,
 * a matching frame is B's no-delta completion and must append as a new
 * occurrence even when its text repeats an earlier answer.
 *
 * Pure, so the gateway handler can classify before it runs turn-end effects.
 */
export function previousTurnFrameIndex(
  state: ClientSessionState | undefined,
  sessionId: string,
  text: string,
  flags: { responsePreviewed?: boolean; responseTransformed?: boolean } = {}
): null | number {
  if (!state?.turnLive || state.interrupted || state.interimBoundaryPending) {
    return null
  }

  const finalText = renderMediaTags(text).trim()

  if (!finalText || flags.responsePreviewed || flags.responseTransformed || !state.streamId) {
    return null
  }

  const messages = state.messages

  // A projected queued prompt is the NEXT turn, not a boundary.
  const newestUserIndex = messages.findLastIndex(
    message => message.role === 'user' && message.id !== `user-queued-${sessionId}`
  )

  const liveIndex = messages.findIndex((message, index) => index > newestUserIndex && message.id === state.streamId)

  if (newestUserIndex < 0 || liveIndex < 0) {
    return null
  }

  const liveText = chatMessageText(messages[liveIndex]).trim()

  // No streamed text yet, or the frame continues it: this turn's own terminal.
  if (!liveText || continuesText(finalText, liveText)) {
    return null
  }

  for (let index = newestUserIndex - 1; index >= 0; index -= 1) {
    const row = messages[index]

    // Interim rows are same-turn seals governed by the boundary-flag paths.
    if (row.role !== 'assistant' || row.hidden || row.interim) {
      continue
    }

    const rowText = chatMessageText(row).trim()

    if (rowText && continuesText(finalText, rowText)) {
      return index
    }
  }

  return null
}
