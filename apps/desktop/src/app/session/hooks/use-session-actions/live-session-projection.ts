import { textWithoutReferenceLines } from '@/components/assistant-ui/reference-kinds'
import { assistantTextPart, type ChatMessage, chatMessageText, textPart, toChatMessages } from '@/lib/chat-messages'
import { parseErrorSurface } from '@/lib/error-surface'
import type { SessionResumeResult } from '@/types/hermes'

import {
  carryRowIdentity,
  committedDuringTurn,
  committedReplyCovers,
  hasStructuralParts,
  isLiveTailRow,
  isPrompt,
  isStrictAnswerTextExtension,
  isSyntheticUserMarker,
  preserveStructuralParts
} from './utils'

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
const safelyUnpersistedInflightUser = Symbol('safelyUnpersistedInflightUser')

type LiveSessionProjection = Pick<SessionResumeResult, 'inflight' | 'queued' | 'session_id' | 'turn_started_at'> & {
  [safelyPersistedInflightUser]?: true
  [safelyUnpersistedInflightUser]?: true
}

type ReconciledSessionResumeResult = SessionResumeResult & {
  [safelyPersistedInflightUser]?: true
  [safelyUnpersistedInflightUser]?: true
}

/** The gateway's backend-clock turn boundary; older gateways omit it. */
export function finiteTurnStartedAt(projection: Pick<SessionResumeResult, 'turn_started_at'>): number | null {
  const startedAt = projection.turn_started_at

  return typeof startedAt === 'number' && Number.isFinite(startedAt) ? startedAt : null
}

/**
 * The authoritative user rows of the running turn, ending at
 * `latestUserIndex`; empty when that row belongs to an earlier turn.
 *
 * A mid-turn redirect gives that turn a RUN of user rows (prompt +
 * corrections). Arrival order seals already-streamed output BETWEEN those
 * rows (#73793), so collect the run by walking back over the live tail:
 * user rows count, live-tail assistant rows are skipped, and a committed
 * assistant reply ends the turn.
 */
function currentTurnUserRun(
  messages: ChatMessage[],
  latestUserIndex: number,
  projection: LiveSessionProjection,
  belongsToCurrentTurn: (message: ChatMessage) => boolean
): ChatMessage[] {
  // A same-text prompt can legitimately start a new turn. If the latest human
  // row is followed by a completed reply from BEFORE this live turn, it is
  // historical and must not suppress the newly accepted inflight copy. Missing
  // timestamps fail toward preserving the prompt: a duplicate can reconcile on
  // the next hydrate, while a dropped accepted turn cannot be recovered.
  const hasPriorCompletedReplyAfterLatestUser =
    latestUserIndex >= 0 &&
    messages
      .slice(latestUserIndex + 1)
      .some(message => message.role === 'assistant' && !isLiveTailRow(message) && !belongsToCurrentTurn(message))

  const latestUserBelongsToCurrentTurn =
    latestUserIndex >= 0 &&
    projection[safelyUnpersistedInflightUser] !== true &&
    (projection[safelyPersistedInflightUser] === true ||
      belongsToCurrentTurn(messages[latestUserIndex]) ||
      !hasPriorCompletedReplyAfterLatestUser)

  const latestUserRun: ChatMessage[] = []

  for (let index = latestUserBelongsToCurrentTurn ? latestUserIndex : -1; index >= 0; index -= 1) {
    const candidate = messages[index]

    if (isSyntheticUserMarker(candidate)) {
      continue
    }

    if (candidate.role === 'user') {
      latestUserRun.unshift(candidate)

      continue
    }

    if (candidate.role === 'assistant' && (isLiveTailRow(candidate) || belongsToCurrentTurn(candidate))) {
      continue
    }

    break
  }

  return latestUserRun
}

/**
 * The inflight prompt projected through the same conversion history uses, so
 * the live bubble matches its persisted twin: attachment refs lift into the
 * chip row, and a synthetic starting prompt (process_complete, hidden, …)
 * takes the display typing its row will get (#112144) — `hidden` yields
 * nothing. Older gateways carry display typing without the provenance flag,
 * so either form takes the persisted-row projection, metadata included.
 */
function inflightPromptRows(projection: LiveSessionProjection, inflightUser: string): ChatMessage[] {
  const inflight = projection.inflight

  if (!inflightUser) {
    return []
  }

  if (inflight?.user_originated !== false && !inflight?.display_kind) {
    return toChatMessages([
      {
        role: 'user',
        content: inflightUser,
        ...(inflight?.user_originated === true ? { user_originated: true } : {})
      }
    ])
  }

  const turnStartedAt = finiteTurnStartedAt(projection)

  return toChatMessages([
    {
      role: 'user',
      content: inflightUser,
      display_kind: inflight?.display_kind,
      display_metadata: inflight?.display_metadata,
      user_originated: inflight?.user_originated,
      ...(turnStartedAt !== null ? { timestamp: turnStartedAt } : {})
    }
  ])
}

/**
 * Where a running runtime wake's notice already sits in `messages`, or -1
 * (always, for a human turn). A runtime wake can itself be the running turn:
 * its visible notice is matched separately so it never occupies a human-turn
 * ordinal.
 */
function currentRuntimeNoticeIndex(
  messages: ChatMessage[],
  runtimeInflight: boolean,
  notice: ChatMessage | undefined,
  belongsToCurrentTurn: (message: ChatMessage) => boolean
): number {
  if (!runtimeInflight || !notice) {
    return -1
  }

  return messages.findLastIndex(
    message =>
      message.role === notice.role &&
      (message.role !== 'user' || message.userOriginated === false) &&
      belongsToCurrentTurn(message) &&
      normalizedMessageText(message) === normalizedMessageText(notice)
  )
}

/**
 * Whether a hydrated assistant row of the running runtime turn stands in for
 * its live reply. A flushed tool-only scaffold precedes the new answer; it is
 * not a cached live reply. Reasoning-bearing rows still suppress flat dumps
 * even before answer text exists (#76444).
 */
function flushedRuntimeReply(message: ChatMessage, belongsToCurrentTurn: (message: ChatMessage) => boolean): boolean {
  return (
    belongsToCurrentTurn(message) &&
    (message.parts.some(part => part.type === 'reasoning') || chatMessageText(message).trim().length > 0)
  )
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
  const latestUserIndex = messages.findLastIndex(isPrompt)
  const turnStartedAt = finiteTurnStartedAt(projection)
  // Hydrated rows do not retain the renderer's `pending` bit, so an assistant
  // already flushed by the CURRENT turn looks settled. The gateway and agent
  // stamp these values from the same wall clock: `turn_started_at` is recorded
  // before run_conversation stamps its user row. Use that boundary only for
  // turn membership, never transcript ordering (SQLite row ids own ordering).
  const belongsToCurrentTurn = (message: ChatMessage) => committedDuringTurn(turnStartedAt, message.timestamp)
  const latestUserRun = currentTurnUserRun(messages, latestUserIndex, projection, belongsToCurrentTurn)

  const persistedInLatestRun = (text: string): boolean =>
    latestUserRun.some(
      message => textWithoutReferenceLines(chatMessageText(message)) === textWithoutReferenceLines(text)
    )

  // A runtime wake can itself be the running turn. Keep its visible notice,
  // but match it separately so it never occupies a human-turn ordinal. Use
  // hydration for typed timeline events, just as the persisted row does.
  const runtimeInflight = projection.inflight?.user_originated === false

  const runtimeBoundary = runtimeInflight && turnStartedAt !== null ? { runtimeTurnStartedAt: turnStartedAt } : {}

  const promptRows = inflightPromptRows(projection, inflightUser)
  const runtimeNoticeIndex = currentRuntimeNoticeIndex(messages, runtimeInflight, promptRows[0], belongsToCurrentTurn)

  const inflightUserAlreadyPersisted = runtimeInflight
    ? runtimeNoticeIndex >= 0
    : projection[safelyPersistedInflightUser] === true || (Boolean(inflightUser) && persistedInLatestRun(inflightUser))

  if (inflightUser && !inflightUserAlreadyPersisted) {
    projected.push(...promptRows.map(message => ({ ...message, ...runtimeBoundary, id: `user-inflight-${sessionId}` })))
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

    // Runtime wakes have their own boundary, independent of human ordinals.
    const turnUserIndex = runtimeInflight ? runtimeNoticeIndex : latestUserIndex

    if (turnUserIndex < 0) {
      // A hidden runtime wake has no displayed user boundary. Backend-clock
      // timestamps still prove which hydrated assistant belongs to its turn.
      return runtimeInflight
        ? (messages.findLast(message => message.role === 'assistant' && belongsToCurrentTurn(message)) ?? null)
        : null
    }

    for (let index = messages.length - 1; index > turnUserIndex; index -= 1) {
      if (messages[index].role === 'assistant') {
        return messages[index]
      }
    }

    return null
  })()

  const turnAlreadyStructured = Boolean(
    liveAssistantOfCurrentTurn &&
    hasStructuralParts(liveAssistantOfCurrentTurn) &&
    (isLiveTailRow(liveAssistantOfCurrentTurn) ||
      (runtimeInflight && flushedRuntimeReply(liveAssistantOfCurrentTurn, belongsToCurrentTurn)))
  )

  // An activate snapshot taken before the turn committed goes stale once REST
  // returns the committed reply after the persisted prompt: its partial
  // `inflight.assistant` is a prefix of that reply, and projecting it paints
  // the answer twice (the extra row frozen on its first chunk). Text alone
  // cannot tell this turn's reply from the previous turn's answer to the same
  // resent prompt, so the reply must have been written after this turn began.
  const committedAt = liveAssistantOfCurrentTurn?.timestamp

  const turnAlreadyCommitted = Boolean(
    inflightUserAlreadyPersisted &&
    !inflightError &&
    liveAssistantOfCurrentTurn &&
    !isLiveTailRow(liveAssistantOfCurrentTurn) &&
    committedDuringTurn(turnStartedAt, committedAt) &&
    committedReplyCovers(liveAssistantOfCurrentTurn, inflightAssistant)
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
      ...runtimeBoundary,
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
          ...runtimeBoundary,
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
      ...runtimeBoundary,
      parts: tail.trim() ? [assistantTextPart(tail)] : [],
      pending: inflightStreaming
    })
  } else {
    if (projectAssistantDump) {
      const liveRow: ChatMessage = {
        id: liveStreamId,
        role: 'assistant',
        ...runtimeBoundary,
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
      parts: [textPart(queuedUser)]
    })
  }

  // Hydrated current-runtime rows can already contain the structured reply
  // that suppresses the flat projection. Mark those rows too: the journal must
  // not walk behind this runtime turn looking for an earlier human prompt.
  let withRuntimeBoundary = messages

  if (runtimeBoundary.runtimeTurnStartedAt !== undefined) {
    withRuntimeBoundary = messages.map((message, index) => {
      if (index !== runtimeNoticeIndex && !belongsToCurrentTurn(message)) {
        return message
      }

      return {
        ...message,
        ...runtimeBoundary,
        ...(message === liveAssistantOfCurrentTurn && turnAlreadyStructured ? { pending: inflightStreaming } : {})
      }
    })
  }

  if (foldTarget) {
    // Splice the folded live row (plus any corrections/queued tail) into the
    // committed partial's slot instead of appending beside it.
    return [
      ...withRuntimeBoundary.slice(0, committedPartialAt),
      ...projected,
      ...withRuntimeBoundary.slice(committedPartialAt + 1)
    ]
  }

  return projected.length ? [...withRuntimeBoundary, ...projected] : withRuntimeBoundary
}

function normalizedMessageText(message: ChatMessage): string {
  return chatMessageText(message).replace(/\s+/g, ' ').trim()
}

function transcriptAnchorMatches(a: ChatMessage, b: ChatMessage): boolean {
  if (a.role !== b.role) {
    return false
  }

  if (a.rowId !== undefined && b.rowId !== undefined) {
    return a.rowId === b.rowId
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
  projection: SessionResumeResult,
  localMessages: ChatMessage[] = []
): ReconciledSessionResumeResult {
  const inflightUser = projection.inflight?.user?.replace(/\s+/g, ' ').trim() ?? ''

  // Modern gateways provide the backend-clock boundary needed to classify
  // hydrated rows in appendLiveSessionProjection. Do not let this older-
  // gateway text fallback override that stronger evidence.
  if (!inflightUser || finiteTurnStartedAt(projection) !== null) {
    return projection
  }

  let suffixStart = 0
  let localLiveIntervalProven = false

  // `omit_messages` is the normal Desktop resume shape, so an older gateway
  // can provide neither runtime history nor turn_started_at. The local view is
  // still useful evidence: removing the exact optimistic-user + stream pair
  // leaves the committed prefix that existed before this live turn. Anchor
  // that prefix in REST and only consider the newer suffix. This is what keeps
  // a historical identical prompt OUT of the dedupe domain.
  const localCommittedPrefix = localMessages.length
    ? removeRepresentedLocalLiveProjection(localMessages, projection)
    : localMessages

  const removedLocalLiveProjection = localCommittedPrefix.length < localMessages.length

  const lastPersistedAnchorIndex = (anchor: ChatMessage) =>
    persistedMessages.findLastIndex(message => transcriptAnchorMatches(message, anchor))

  if (runtimeMessages.length) {
    const persistedAnchorIndex = lastPersistedAnchorIndex(runtimeMessages[runtimeMessages.length - 1])

    if (persistedAnchorIndex < 0) {
      return projection
    }

    suffixStart = persistedAnchorIndex + 1
    localLiveIntervalProven = removedLocalLiveProjection
  } else if (removedLocalLiveProjection) {
    // Renderer-owned optimistic/projection rows and synthetic role=user
    // notices cannot prove a durable prefix boundary. Walk backward over the
    // remaining local transcript until a real persisted row maps into REST.
    const anchorCandidates = localCommittedPrefix.filter(
      message =>
        !isSyntheticUserMarker(message) &&
        (!message.id.startsWith('user-') || message.rowId !== undefined || message.timestamp !== undefined) &&
        !isLiveTailRow(message)
    )

    let persistedAnchorIndex = -1

    for (let localIndex = anchorCandidates.length - 1; localIndex >= 0 && persistedAnchorIndex < 0; localIndex -= 1) {
      persistedAnchorIndex = lastPersistedAnchorIndex(anchorCandidates[localIndex])
    }

    if (persistedAnchorIndex < 0 && anchorCandidates.length) {
      return projection
    }

    suffixStart = persistedAnchorIndex + 1
    localLiveIntervalProven = true
  }

  const persistedTail = persistedMessages.slice(suffixStart)
  const lastPersistedMessage = persistedTail[persistedTail.length - 1]
  const latestHumanUserIndex = persistedTail.findLastIndex(isPrompt)

  // On old gateways the local pair can prove where the live interval starts,
  // but it cannot prove that a same-text REST turn inside that interval is the
  // newly accepted one. A completed assistant after that user may belong to an
  // older repeated prompt missing from the stale local prefix. Fail toward
  // preserving the optimistic prompt; modern gateways disambiguate current
  // assistant rows with turn_started_at in appendLiveSessionProjection.
  const latestHumanUserHasCompletedReply =
    latestHumanUserIndex >= 0 &&
    persistedTail
      .slice(latestHumanUserIndex + 1)
      .some(message => message.role === 'assistant' && !isLiveTailRow(message))

  const persistedUserPresent = localLiveIntervalProven
    ? latestHumanUserIndex >= 0 &&
      normalizedMessageText(persistedTail[latestHumanUserIndex]) === inflightUser &&
      !latestHumanUserHasCompletedReply
    : lastPersistedMessage?.role === 'user' && normalizedMessageText(lastPersistedMessage) === inflightUser

  if (!persistedUserPresent) {
    return localLiveIntervalProven ? { ...projection, [safelyUnpersistedInflightUser]: true } : projection
  }

  return { ...projection, [safelyPersistedInflightUser]: true }
}

/**
 * Whether a local optimistic-user + stream pair is the running turn's
 * projection. The activation snapshot and the local stream are read at
 * different times while the turn keeps streaming in the background, so
 * neither is guaranteed to be textually identical to the other even though
 * both represent the same running reply — one is simply further along.
 */
function representsLivePair(
  user: ChatMessage,
  assistant: ChatMessage | undefined,
  inflightUser: string,
  inflightAssistant: string
): boolean {
  const assistantText = assistant ? normalizedMessageText(assistant) : ''

  return (
    user.role === 'user' &&
    user.id.startsWith('user-') &&
    normalizedMessageText(user) === inflightUser &&
    assistant?.role === 'assistant' &&
    assistant.id.startsWith('assistant-stream-') &&
    (assistantText === inflightAssistant ||
      isStrictAnswerTextExtension(inflightAssistant, assistantText) ||
      isStrictAnswerTextExtension(assistantText, inflightAssistant))
  )
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

  let inflightUserIndex = -1

  // Repeated prompts are valid. Match the newest complete optimistic-user +
  // stream boundary rather than the first same-text user in the open tail;
  // an older interrupted prompt may otherwise shadow the live pair.
  for (let index = previousMessages.length - 2; index >= openTailStart; index -= 1) {
    if (representsLivePair(previousMessages[index], previousMessages[index + 1], inflightUser, inflightAssistant)) {
      inflightUserIndex = index

      break
    }
  }

  if (inflightUserIndex < 0) {
    return previousMessages
  }

  const assistantIndex = inflightUserIndex + 1

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

/**
 * Both resume paths pair a running turn's snapshot with the local view the
 * same way: drop the local live rows the snapshot replaces, and check its
 * inflight prompt against the persisted transcript (the local view proves the
 * live interval when an older gateway omits runtime history).
 */
export function reconcileLocalLiveProjection(
  localMessages: ChatMessage[],
  persistedMessages: ChatMessage[],
  projection: SessionResumeResult
): { liveProjection: ReconciledSessionResumeResult; previousMessages: ChatMessage[] } {
  return {
    liveProjection: dedupeInflightUserAgainstTranscript(
      persistedMessages,
      toChatMessages(projection.messages),
      projection,
      localMessages
    ),
    previousMessages: removeRepresentedLocalLiveProjection(localMessages, projection)
  }
}
