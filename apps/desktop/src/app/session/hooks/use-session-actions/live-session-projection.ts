import { textWithoutReferenceLines } from '@/components/assistant-ui/reference-kinds'
import {
  assistantTextPart,
  type ChatMessage,
  chatMessageText,
  clientMessageIdFromMetadata,
  sameClientIdentity,
  textPart,
  toChatMessages
} from '@/lib/chat-messages'
import { parseErrorSurface } from '@/lib/error-surface'
import type { SessionResumeResult } from '@/types/hermes'

import {
  carryRowIdentity,
  hasStructuralParts,
  isLiveTailRow,
  isStrictAnswerTextExtension,
  preserveStructuralParts
} from './utils'

/**
 * True only when a row was written after the live turn began. Text alone
 * cannot tell this turn's reply from the previous turn's answer to the same
 * resent prompt, so a missing timestamp (older runtime) never passes.
 */
function committedDuringTurn(turnStartedAt: unknown, committedAt: unknown): boolean {
  return typeof turnStartedAt === 'number' && typeof committedAt === 'number' && committedAt >= turnStartedAt
}

/**
 * Append the backend-only tail of a live turn to a stored transcript.
 *
 * Session history is committed only when a turn finishes. During a reconnect,
 * `inflight` is therefore the authority for the currently running user/assistant
 * pair, while `queued` is an accepted next-turn prompt waiting in gateway
 * memory. Stable ids let repeated activate/resume hydration reconcile instead
 * of growing duplicate rows.
 */
const safelyPersistedInflightUser = Symbol('safelyPersistedInflightUser')

type LiveSessionProjection = Pick<SessionResumeResult, 'inflight' | 'queued' | 'session_id' | 'turn_started_at'> & {
  [safelyPersistedInflightUser]?: true
}

type ReconciledSessionResumeResult = SessionResumeResult & {
  [safelyPersistedInflightUser]?: true
}

export function appendLiveSessionProjection(messages: ChatMessage[], projection: LiveSessionProjection): ChatMessage[] {
  const inflightUser = projection.inflight?.user?.trim() ?? ''
  const inflightAssistant = projection.inflight?.assistant ?? ''
  const inflightStreaming = Boolean(projection.inflight?.streaming)

  // Mid-turn redirect corrections. They are additional user bubbles belonging
  // to this same turn, ordered by arrival: after the output that had already
  // streamed when they were typed, before the output they redirected.
  // `correction_offsets` (assistant-text length at each accepted correction)
  // carries that boundary; older gateways omit it.
  const rawCorrections = projection.inflight?.corrections ?? []
  const rawOffsets = projection.inflight?.correction_offsets

  const inflightCorrectionEntries = rawCorrections
    .map((correction, index) => ({ text: correction?.trim() ?? '', offset: rawOffsets?.[index] }))
    .filter(entry => entry.text)

  const inflightCorrections = inflightCorrectionEntries.map(entry => entry.text)

  const correctionOffsetsUsable =
    inflightCorrectionEntries.length > 0 &&
    inflightCorrectionEntries.every(entry => typeof entry.offset === 'number' && entry.offset >= 0)

  // A retained failed turn (the gateway keeps error snapshots replayable when
  // the terminal frame may have been lost to a disconnect) — surface the
  // failure on the projected row instead of rendering the partial as healthy.
  const inflightError = projection.inflight?.error?.trim() ?? ''
  const inflightErrorSurface = parseErrorSurface(projection.inflight?.error_surface)
  const queuedUser = projection.queued?.user?.trim() ?? ''

  if (
    !inflightUser &&
    !inflightAssistant &&
    !inflightStreaming &&
    !inflightError &&
    !queuedUser &&
    !inflightCorrections.length
  ) {
    return messages
  }

  const sessionId = projection.session_id || 'session'
  const projected: ChatMessage[] = []
  // A turn normally persists its user row before inference begins. session.resume
  // then returns that stored row *and* the still-live inflight projection; adding
  // both makes a backgrounded prompt appear twice when its session is reopened.
  // Only suppress the projection when the latest authoritative user row is the
  // same turn — older identical prompts must not hide a newly accepted repeat.
  // A mid-turn redirect gives that turn a RUN of user rows (prompt +
  // corrections). Arrival order seals already-streamed output BETWEEN those
  // rows (#73793), so collect the run by walking back over the live tail:
  // user rows count, live-tail assistant rows are skipped, and a committed
  // assistant reply ends the turn.
  const latestUserIndex = messages.map(message => message.role).lastIndexOf('user')
  const latestUserRun: ChatMessage[] = []

  for (let index = latestUserIndex; index >= 0; index -= 1) {
    const candidate = messages[index]

    if (candidate.role === 'user') {
      latestUserRun.unshift(candidate)

      continue
    }

    if (candidate.role === 'assistant' && isLiveTailRow(candidate)) {
      continue
    }

    break
  }

  const inflightClientMessageId =
    projection.inflight?.client_message_id || clientMessageIdFromMetadata(projection.inflight?.display_metadata)

  const persistedInLatestRun = (text: string): boolean =>
    latestUserRun.some(
      message =>
        sameClientIdentity({ clientMessageId: inflightClientMessageId }, message) ??
        textWithoutReferenceLines(chatMessageText(message)) === textWithoutReferenceLines(text)
    )

  const inflightUserAlreadyPersisted =
    projection[safelyPersistedInflightUser] === true || (Boolean(inflightUser) && persistedInLatestRun(inflightUser))

  if (inflightUser && !inflightUserAlreadyPersisted) {
    // Project the prompt through the same conversion history uses, so the live
    // bubble matches its persisted twin: attachment refs lift into the chip row,
    // and a synthetic starting prompt (process_complete, hidden, …) takes the
    // display typing its row will get (#112144) — `hidden` yields nothing.
    const displayKind = projection.inflight?.display_kind

    const typed = toChatMessages([
      {
        role: 'user',
        content: inflightUser,
        ...(displayKind ? { display_kind: displayKind } : {}),
        ...(projection.inflight?.display_metadata !== undefined
          ? { display_metadata: projection.inflight.display_metadata }
          : {})
      }
    ])

    projected.push(
      ...typed.map(message => ({
        ...message,
        id: `user-inflight-${sessionId}`,
        ...(inflightClientMessageId ? { clientMessageId: inflightClientMessageId } : {}),
        ...(projection.inflight?.user_timestamp !== undefined ? { timestamp: projection.inflight.user_timestamp } : {})
      }))
    )
  }

  // Keep a pending assistant boundary even before the first delta when a
  // queued user turn follows it. This preserves the two distinct turns.
  //
  // When the *current live turn* already holds a structured mid-turn assistant
  // row (reasoning / tool-call from the live stream or journal), do NOT append
  // a pure-text projection of `inflight.assistant` — that flat dump re-renders
  // thinking as answer text and sandwiches the structured parts (#76444).
  // Only inspect the live tail after the latest user run — never a completed
  // historical tool-bearing reply earlier in the transcript (review feedback).
  const liveStreamId = `assistant-stream-${sessionId}`

  const liveAssistantOfCurrentTurn = ((): ChatMessage | null => {
    const byStreamId = messages.find(message => message.id === liveStreamId)

    if (byStreamId) {
      return byStreamId
    }

    // Assistants after the latest user row belong to this turn's tail.
    if (latestUserIndex < 0) {
      return null
    }

    for (let index = messages.length - 1; index > latestUserIndex; index -= 1) {
      if (messages[index].role === 'assistant') {
        return messages[index]
      }
    }

    return null
  })()

  const turnAlreadyStructured = Boolean(
    liveAssistantOfCurrentTurn &&
    hasStructuralParts(liveAssistantOfCurrentTurn) &&
    isLiveTailRow(liveAssistantOfCurrentTurn)
  )

  // An activate snapshot taken before the turn committed goes stale once REST
  // returns the committed reply after the persisted prompt: its partial
  // `inflight.assistant` is a prefix of that reply, and projecting it paints
  // the answer twice (the extra row frozen on its first chunk). Text alone
  // cannot tell this turn's reply from the previous turn's answer to the same
  // resent prompt, so the reply must have been written after this turn began.
  const turnStartedAt = projection.turn_started_at
  const committedAt = liveAssistantOfCurrentTurn?.timestamp

  const turnAlreadyCommitted = Boolean(
    inflightUserAlreadyPersisted &&
    !inflightError &&
    liveAssistantOfCurrentTurn &&
    !isLiveTailRow(liveAssistantOfCurrentTurn) &&
    committedDuringTurn(turnStartedAt, committedAt) &&
    !liveAssistantOfCurrentTurn.parts.some(part => part.type === 'tool-call') &&
    isStrictAnswerTextExtension(chatMessageText(liveAssistantOfCurrentTurn), inflightAssistant)
  )

  const wantsAssistantRow = Boolean(
    inflightAssistant || inflightStreaming || inflightError || (inflightUser && queuedUser)
  )

  const projectAssistantDump = wantsAssistantRow && !turnAlreadyCommitted && !(turnAlreadyStructured && !inflightError)

  // #121122, the mirror of turnAlreadyCommitted: REST already holds this
  // turn's PARTIAL assistant row (committed as the turn progressed, tool
  // blocks included) while `inflight` still streams the fuller dump.
  // Appending the dump paints the turn twice — the frozen partial with its
  // action bar plus the live copy repeating it. Fold the dump into the tail
  // row instead: same live id (deltas keep landing), fuller text, committed
  // structure carried over. Only when the dump extends the tail text — a
  // diverged tail is a different reply and both rows survive.
  const committedPartial =
    liveAssistantOfCurrentTurn && !isLiveTailRow(liveAssistantOfCurrentTurn) ? liveAssistantOfCurrentTurn : null

  const committedPartialAt = committedPartial ? messages.lastIndexOf(committedPartial) : -1

  const committedPartialText = committedPartial ? chatMessageText(committedPartial) : ''

  const turnPartiallyCommitted = Boolean(
    projectAssistantDump &&
    !inflightError &&
    inflightStreaming &&
    committedPartial &&
    inflightUserAlreadyPersisted &&
    !correctionOffsetsUsable &&
    committedDuringTurn(turnStartedAt, committedAt) &&
    (committedPartialText.trim() === inflightAssistant.trim() ||
      isStrictAnswerTextExtension(inflightAssistant, committedPartialText))
  )

  const foldTarget = turnPartiallyCommitted ? committedPartial : null

  const pushCorrection = (correction: string, index: number): void => {
    if (persistedInLatestRun(correction)) {
      return
    }

    projected.push({
      id: `user-inflight-correction-${index}-${sessionId}`,
      role: 'user',
      parts: [textPart(correction)]
    })
  }

  // Corrections typed while the turn ran are ordered by ARRIVAL: each lands
  // after the assistant output that had already streamed when it was typed and
  // before the output it redirected (#73793 — the old prompt → corrections →
  // reply order spliced them above screens of output the user had already
  // read). With usable offsets the flat dump is split at each boundary; without
  // them (older gateway, or a structured/error tail that must stay whole) the
  // corrections follow the projected reply, matching the live transcript's
  // append-at-tail contract.
  if (projectAssistantDump && correctionOffsetsUsable && !inflightError && inflightAssistant) {
    let cursor = 0

    for (const [index, entry] of inflightCorrectionEntries.entries()) {
      const boundary = Math.min(Math.max(entry.offset as number, cursor), inflightAssistant.length)
      const segment = inflightAssistant.slice(cursor, boundary)

      if (segment.trim()) {
        // Sealed pre-correction output. The `inflight-assistant-` prefix marks
        // it a live-tail row so repeated resumes keep the user run intact.
        projected.push({
          id: `inflight-assistant-segment-${index}-${sessionId}`,
          role: 'assistant',
          parts: [assistantTextPart(segment)],
          pending: false,
          interim: true
        })
      }

      cursor = boundary
      pushCorrection(entry.text, index)
    }

    const tail = inflightAssistant.slice(cursor)

    projected.push({
      id: liveStreamId,
      role: 'assistant',
      parts: tail.trim() ? [assistantTextPart(tail)] : [],
      pending: inflightStreaming
    })
  } else {
    if (projectAssistantDump) {
      const liveRow: ChatMessage = {
        id: liveStreamId,
        role: 'assistant',
        parts: inflightAssistant ? [assistantTextPart(inflightAssistant)] : [],
        pending: inflightStreaming,
        ...(inflightError ? { error: inflightError } : {}),
        ...(inflightError && inflightErrorSurface ? { errorSurface: inflightErrorSurface } : {})
      }

      if (foldTarget) {
        // #121122: the persisted tail IS this turn's partial — replace it in
        // place so the transcript holds one row that keeps streaming.
        // Structure the flat dump cannot express (tool calls, reasoning)
        // carries over; row id and reactions stay so nothing blinks off
        // mid-turn and a reaction toggle still reaches the persisted row.
        projected.push(carryRowIdentity(preserveStructuralParts(liveRow, foldTarget), foldTarget))
      } else {
        projected.push(liveRow)
      }
    }

    for (const [index, correction] of inflightCorrections.entries()) {
      pushCorrection(correction, index)
    }
  }

  if (queuedUser) {
    projected.push({
      id: `user-queued-${sessionId}`,
      role: 'user',
      parts: [textPart(queuedUser)],
      ...(projection.queued?.client_message_id ? { clientMessageId: projection.queued.client_message_id } : {}),
      ...(projection.queued?.user_timestamp !== undefined ? { timestamp: projection.queued.user_timestamp } : {})
    })
  }

  if (foldTarget) {
    // Splice the folded live row (plus any corrections/queued tail) into the
    // committed partial's slot instead of appending beside it.
    return [...messages.slice(0, committedPartialAt), ...projected, ...messages.slice(committedPartialAt + 1)]
  }

  return projected.length ? [...messages, ...projected] : messages
}

function normalizedMessageText(message: ChatMessage): string {
  return chatMessageText(message).replace(/\s+/g, ' ').trim()
}

function transcriptAnchorMatches(a: ChatMessage, b: ChatMessage): boolean {
  if (a.role !== b.role) {
    return false
  }

  const aText = normalizedMessageText(a)
  const bText = normalizedMessageText(b)

  if (a.timestamp !== undefined && b.timestamp !== undefined) {
    return a.timestamp === b.timestamp && aText === bText
  }

  return Boolean(aText) && aText === bText
}

/**
 * Mark only an already-materialized `inflight.user` for visual suppression.
 *
 * A running gateway returns two independent truths: its compressed runtime
 * history plus the current in-flight turn, while REST may already have flushed
 * that user row into the complete persisted transcript. Global text dedupe is
 * unsafe because users may intentionally submit the same prompt twice. Instead,
 * find the last runtime message inside the persisted transcript and inspect only
 * the newer persisted suffix.
 *
 * Keep `inflight.user` intact because it also carries turn structure: a queued
 * prompt needs its assistant boundary even when the persisted user has no
 * assistant delta yet. The private marker lets the renderer suppress only that
 * duplicate bubble. If the histories have no safe common anchor, keep the
 * projection unchanged — a duplicate is recoverable, but dropping a real
 * accepted prompt is not.
 */
export function dedupeInflightUserAgainstTranscript(
  persistedMessages: ChatMessage[],
  runtimeMessages: ChatMessage[],
  projection: SessionResumeResult
): ReconciledSessionResumeResult {
  const inflightUser = projection.inflight?.user?.replace(/\s+/g, ' ').trim() ?? ''

  if (!inflightUser) {
    return projection
  }

  let suffixStart = 0

  if (runtimeMessages.length) {
    const runtimeAnchor = runtimeMessages[runtimeMessages.length - 1]
    let persistedAnchorIndex = -1

    for (let index = persistedMessages.length - 1; index >= 0; index -= 1) {
      if (transcriptAnchorMatches(persistedMessages[index], runtimeAnchor)) {
        persistedAnchorIndex = index

        break
      }
    }

    if (persistedAnchorIndex < 0) {
      return projection
    }

    suffixStart = persistedAnchorIndex + 1
  }

  const persistedTail = persistedMessages.slice(suffixStart)
  const lastPersistedMessage = persistedTail[persistedTail.length - 1]

  const persistedUserPresent =
    lastPersistedMessage?.role === 'user' &&
    (sameClientIdentity({ clientMessageId: projection.inflight?.client_message_id }, lastPersistedMessage) ??
      normalizedMessageText(lastPersistedMessage) === inflightUser)

  if (!persistedUserPresent) {
    return projection
  }

  return { ...projection, [safelyPersistedInflightUser]: true }
}

/**
 * Drop only synthetic local tail rows that the activation snapshot replaces.
 * Unmatched optimistic rows survive so a submit racing with activation is not
 * lost; completed transcript rows before the open tail are never considered.
 */
export function removeRepresentedLocalLiveProjection(
  previousMessages: ChatMessage[],
  projection: Pick<SessionResumeResult, 'inflight' | 'queued'>
): ChatMessage[] {
  const inflightUser = projection.inflight?.user?.replace(/\s+/g, ' ').trim() ?? ''
  const inflightAssistant = projection.inflight?.assistant?.replace(/\s+/g, ' ').trim() ?? ''
  const queuedUser = projection.queued?.user?.replace(/\s+/g, ' ').trim() ?? ''

  const hasAssistantProjection = Boolean(
    projection.inflight?.assistant || projection.inflight?.streaming || (inflightUser && queuedUser)
  )

  if (!inflightUser || !hasAssistantProjection) {
    return previousMessages
  }

  let openTailStart = 0

  for (let index = previousMessages.length - 1; index >= 0; index -= 1) {
    const message = previousMessages[index]

    if (message.role === 'assistant' && !message.pending) {
      openTailStart = index + 1

      break
    }
  }

  const inflightUserIndex = previousMessages.findIndex(
    (message, index) =>
      index >= openTailStart &&
      message.role === 'user' &&
      message.id.startsWith('user-') &&
      normalizedMessageText(message) === inflightUser
  )

  const assistantIndex = inflightUserIndex + 1
  const assistant = previousMessages[assistantIndex]
  const localAssistant = assistant ? normalizedMessageText(assistant) : ''

  // The activation snapshot and this local row are read at different times
  // while the turn keeps streaming in the background, so neither is
  // guaranteed to be textually identical to the other even though both
  // represent the same running reply — one is simply further along.
  const assistantTextRepresented =
    localAssistant === inflightAssistant ||
    isStrictAnswerTextExtension(localAssistant, inflightAssistant) ||
    isStrictAnswerTextExtension(inflightAssistant, localAssistant)

  const assistantMatches =
    inflightUserIndex >= openTailStart &&
    assistant?.role === 'assistant' &&
    assistant.id.startsWith('assistant-stream-') &&
    assistantTextRepresented

  if (!assistantMatches) {
    return previousMessages
  }

  let queuedUserIndex = -1

  if (queuedUser) {
    queuedUserIndex = previousMessages.findIndex(
      (message, index) =>
        index > assistantIndex &&
        message.role === 'user' &&
        message.id.startsWith('user-queued-') &&
        normalizedMessageText(message) === queuedUser
    )
  }

  return previousMessages.filter(
    (_message, index) => index !== inflightUserIndex && index !== assistantIndex && index !== queuedUserIndex
  )
}
