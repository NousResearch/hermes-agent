import { textWithoutReferenceLines } from '@/components/assistant-ui/reference-kinds'
import { getSession } from '@/hermes'
import { assistantTextPart, type ChatMessage, chatMessageText, textPart } from '@/lib/chat-messages'
import { normalizePersonalityValue } from '@/lib/chat-runtime'
import { embeddedImageUrls, textWithoutEmbeddedImages } from '@/lib/embedded-images'
import { reconcileApprovalModeForProfile } from '@/store/approval-mode'
import { requestDesktopOnboardingForCredentialWarning } from '@/store/onboarding'
import { $activeGatewayProfile, $profiles, normalizeProfileKey } from '@/store/profile'
import {
  $currentCwd,
  $sessions,
  commitWorkspaceCwdForSelectedSession,
  releaseWorkspaceCwdOwner,
  sessionMatchesStoredId,
  setCurrentBranch,
  setCurrentCwdTransient,
  setCurrentFastMode,
  setCurrentModel,
  setCurrentPersonality,
  setCurrentProvider,
  setCurrentReasoningEffort,
  setCurrentServiceTier,
  setCurrentUsage,
  setSessions,
  setWorkspaceCwdOwner,
  setYoloActive
} from '@/store/session'

// Re-exported for the many session-actions/tile call sites that already import
// it from here; the canonical definition lives in @/store/session.
export { sessionMatchesStoredId }
import { reportBackendContract, reportInstallMethodWarning } from '@/store/updates'
import type { SessionCreateResponse, SessionInfo, SessionResumeResponse, SessionRuntimeInfo } from '@/types/hermes'

import type { ClientSessionState } from '../../../types'

function withAppendedText(message: ChatMessage, suffix: string): ChatMessage {
  let appended = false

  const parts = message.parts.map(part => {
    if (part.type !== 'text' || appended) {
      return part
    }

    appended = true

    return { ...part, text: `${part.text}${suffix}` }
  })

  return appended ? { ...message, parts } : message
}

/** Reasoning / tool-call parts that the gateway inflight dump cannot express. */
function hasStructuralParts(message: ChatMessage): boolean {
  return message.parts.some(part => part.type === 'reasoning' || part.type === 'tool-call')
}

/**
 * A live-turn row — the gateway's text-only `inflight` projection, a
 * still-streaming local bubble, or an interim row sealed inside the running
 * turn — as opposed to a committed transcript row.
 */
function isLiveTailRow(message: ChatMessage): boolean {
  return (
    message.pending === true ||
    message.id.startsWith('assistant-stream-') ||
    message.id.startsWith('inflight-assistant-') ||
    message.interim === true
  )
}

/**
 * True when `next` is a pure forward extension of the previous *answer* text.
 * Empty previous answer never accepts a dump as an extension — that is how the
 * mid-turn inflight flat dump used to sandwich structured rows (#76444).
 */
export function isStrictAnswerTextExtension(next: string, previous: string): boolean {
  const n = next.trim()
  const p = previous.trim()

  if (!p || !n) {
    return false
  }

  return n.startsWith(p)
}

/**
 * Carry structural parts an authoritative row cannot express.
 *
 * A live turn's authoritative projection is TEXT-ONLY: the gateway's `inflight`
 * snapshot carries `user`/`assistant` strings, and history is not committed
 * until the turn finishes. The renderer's cached state is therefore the sole
 * carrier of the running turn's reasoning and tool calls, so switching threads
 * mid-turn and back re-hydrated an assistant row stripped of both — the turn
 * looked inert, with no thinking trace and no tool activity.
 *
 * Preserved only when the rows are the SAME turn: identical text, or the
 * authoritative text extending the cached one (another delta landed). Anything
 * else may be a different turn at the same role ordinal — compression rewrites
 * history — and must not inherit foreign parts. Tool calls dedupe on
 * `toolCallId` so a row that already carries them is left alone.
 */
function preserveStructuralParts(message: ChatMessage, previous: ChatMessage): ChatMessage {
  const carried = previous.parts.filter(part => part.type === 'reasoning' || part.type === 'tool-call')

  if (!carried.length) {
    return message
  }

  const hasReasoning = message.parts.some(part => part.type === 'reasoning')

  const presentToolCallIds = new Set(
    message.parts.flatMap(part => (part.type === 'tool-call' ? [part.toolCallId] : []))
  )

  const missing = carried.filter(part =>
    part.type === 'reasoning' ? !hasReasoning : !presentToolCallIds.has(part.toolCallId)
  )

  return missing.length ? { ...message, parts: [...missing, ...message.parts] } : message
}

// Compile-time exhaustiveness guards. If a new field is added to ChatMessage
// or a new part type appears in the ChatMessagePart union (e.g. @assistant-ui
// ships one), these fail tsc until someone explicitly classifies it.
//
// COMPARED: fields whose change must trigger a re-render (setMessages).
// IGNORED:  fields that are intentionally not compared — display-only metadata
//           or reference identity the runtime already guarantees.
//   timestamp  — presentation-only (sort/age display), never affects transcript equality
//   attachmentRefs — composer-side metadata; already reconciled in reconcileResumeMessages
//   rowId — durable backend identity; stable for a given row, never changes what's painted
//
// If your new field affects what the user sees in the transcript, add it to
// COMPARED. If it's metadata that shouldn't trigger a re-render, add it to
// IGNORED.
const _chatMessageFieldsExhaustive: {
  [K in Exclude<keyof ChatMessage, (typeof COMPARED_FIELDS)[number] | (typeof IGNORED_FIELDS)[number]>]: never
} = {}

const COMPARED_FIELDS = ['id', 'role', 'pending', 'error', 'hidden', 'branchGroupId', 'interim', 'reactions'] as const

const IGNORED_FIELDS = ['timestamp', 'attachmentRefs', 'parts', 'rowId'] as const

// Compile-time check: every ChatMessagePart discriminant must be handled by
// chatPartsEquivalent. If @assistant-ui adds a new part type, this fails tsc.
//   text, reasoning      → compared by .text
//   tool-call             → compared by toolCallId/toolName + result presence
//   source, image, file, data, generative-ui, audio, data-* → shallow primitive compare
const _chatMessagePartTypesExhaustive: {
  [T in Exclude<ChatMessage['parts'][number]['type'], (typeof HANDLED_PART_TYPES)[number]>]: never
} = {}

const HANDLED_PART_TYPES = [
  'text',
  'reasoning',
  'tool-call',
  'source',
  'image',
  'file',
  'data',
  'generative-ui',
  'audio'
] as const

// Structural compare WITHOUT JSON.stringify — the only consumer asks "did
// the transcript change, should I call setMessages?", so a slightly
// conservative compare (occasionally false-negative → one extra idempotent
// setMessages) is safe, but a false-POSITIVE (claiming equal when different)
// would skip a needed update.
export function chatPartsEquivalent(aPart: ChatMessage['parts'][number], bPart: ChatMessage['parts'][number]): boolean {
  // Reference equality fast-path
  if (aPart === bPart) {
    return true
  }

  if (aPart.type !== bPart.type) {
    return false
  }

  if (aPart.type === 'text' || aPart.type === 'reasoning') {
    return (aPart as { text: string }).text === (bPart as { text: string }).text
  }

  if (aPart.type === 'tool-call') {
    const aCall = aPart as { toolCallId?: string; toolName?: string; result?: unknown }
    const bCall = bPart as { toolCallId?: string; toolName?: string; result?: unknown }

    if (aCall.toolCallId !== bCall.toolCallId || aCall.toolName !== bCall.toolName) {
      return false
    }

    // Compare whether result is present (undefined on both or defined on both)
    const aHasResult = aCall.result !== undefined
    const bHasResult = bCall.result !== undefined

    return aHasResult === bHasResult
  }

  // For all other handled part types (source, image, file, data, generative-ui,
  // audio, data-*), fall back to shallow primitive-key comparison — conservative:
  // if we're not sure, claim not-equal (one extra setMessages is harmless, but
  // skipping an update would break the UI).
  const aPrimitive = aPart as Record<string, unknown>
  const bPrimitive = bPart as Record<string, unknown>
  const aKeys = Object.keys(aPrimitive).filter(k => typeof aPrimitive[k] !== 'object' || aPrimitive[k] === null)
  const bKeys = Object.keys(bPrimitive).filter(k => typeof bPrimitive[k] !== 'object' || bPrimitive[k] === null)

  if (aKeys.length !== bKeys.length) {
    return false
  }

  return aKeys.every(k => aPrimitive[k] === bPrimitive[k])
}

export function chatReactionsEquivalent(a: ChatMessage['reactions'], b: ChatMessage['reactions']): boolean {
  const aList = a ?? []
  const bList = b ?? []

  if (aList === bList) {
    return true
  }

  return (
    aList.length === bList.length &&
    aList.every((reaction, index) => reaction.emoji === bList[index].emoji && reaction.author === bList[index].author)
  )
}

export function chatMessagesEquivalent(a: ChatMessage, b: ChatMessage): boolean {
  if (
    a.id !== b.id ||
    a.role !== b.role ||
    a.pending !== b.pending ||
    a.error !== b.error ||
    a.hidden !== b.hidden ||
    a.branchGroupId !== b.branchGroupId ||
    // Interim gates the action footer, so flipping it must repaint (e.g. a
    // previewed final settling onto a sealed interim bubble restores the bar).
    (a.interim ?? false) !== (b.interim ?? false) ||
    !chatReactionsEquivalent(a.reactions, b.reactions)
  ) {
    return false
  }

  if (a.parts.length !== b.parts.length) {
    return false
  }

  return a.parts.every((part, index) => chatPartsEquivalent(part, b.parts[index]))
}

export function chatMessageArraysEquivalent(a: ChatMessage[], b: ChatMessage[]): boolean {
  // Array-level identity fast-path (same reference)
  if (a === b) {
    return true
  }

  return a.length === b.length && a.every((message, index) => chatMessagesEquivalent(message, b[index]))
}

export function reconcileResumeMessages(nextMessages: ChatMessage[], previousMessages: ChatMessage[]): ChatMessage[] {
  if (!previousMessages.length) {
    return nextMessages
  }

  const previousByRoleOrdinal = new Map<string, ChatMessage>()
  const previousRoleCounts = new Map<string, number>()

  for (const message of previousMessages) {
    const ordinal = previousRoleCounts.get(message.role) ?? 0
    previousRoleCounts.set(message.role, ordinal + 1)
    previousByRoleOrdinal.set(`${message.role}:${ordinal}`, message)
  }

  const nextRoleCounts = new Map<string, number>()

  return nextMessages.map(message => {
    const ordinal = nextRoleCounts.get(message.role) ?? 0
    nextRoleCounts.set(message.role, ordinal + 1)

    const previous = previousByRoleOrdinal.get(`${message.role}:${ordinal}`)

    if (!previous) {
      return message
    }

    const nextText = chatMessageText(message).trim()
    const previousText = chatMessageText(previous)
    const previousVisibleText = textWithoutEmbeddedImages(previousText)
    const previousTrimmed = previousVisibleText.trim()
    let preserved = message

    // #75825: resume can project an empty (or lagging) inflight assistant shell
    // at the same role-ordinal as the live stream row that still holds the
    // streamed text, reasoning and tool calls. Prefer that richer pending row
    // instead of painting the shell — otherwise the reply vanishes until
    // restart. Guarded to the same reply further along (see
    // localPendingSupersedes) so a different turn at the same ordinal cannot
    // hijack the slot.
    if (localPendingSupersedes(previous, message)) {
      return withAuthoritativeTurnState(previous, message)
    }

    const sameText = nextText === previousVisibleText || nextText === previousText.trim()

    // Mid-turn, the authoritative text has advanced past the cached copy by one
    // or more deltas. That is still the same turn, and the cached row holds the
    // only copy of its reasoning / tool calls, so treat an extension as a match
    // for structural carry-over. Attachment refs and image re-appending stay on
    // the strict equality path — they reconcile a SETTLED row, and a growing
    // row is by definition not settled.
    //
    // Live-tail identity: structure-only same-turn carry is allowed only when
    // the *structure-bearing cached row* is still the in-flight stream
    // (pending / stream id / interim). Marking only the text-only next row
    // live is not enough — after compression a new live assistant can share a
    // role ordinal with an unrelated historical structured row and must not
    // inherit its reasoning/tool parts (#76444 review / salvage).
    const sameTurn =
      sameText ||
      (nextText.length > 0 && previousTrimmed.length > 0 && isStrictAnswerTextExtension(nextText, previousTrimmed)) ||
      (message.role === 'assistant' &&
        previous.role === 'assistant' &&
        hasStructuralParts(previous) &&
        !hasStructuralParts(message) &&
        isLiveTailRow(previous))

    if (sameTurn) {
      preserved = preserveStructuralParts(preserved, previous)

      // Never replace structured answer text with a non-extending flat dump.
      if (
        message.role === 'assistant' &&
        hasStructuralParts(previous) &&
        !hasStructuralParts(message) &&
        !isStrictAnswerTextExtension(nextText, previousVisibleText)
      ) {
        const nonText = preserved.parts.filter(part => part.type !== 'text')
        const priorAnswer = previous.parts.filter(part => part.type === 'text')
        preserved = { ...preserved, parts: [...nonText, ...priorAnswer] }
      }
    }

    if (
      sameText &&
      message.role === 'user' &&
      preserved.attachmentRefs === undefined &&
      previous.attachmentRefs?.length
    ) {
      preserved = { ...preserved, attachmentRefs: [...previous.attachmentRefs] }
    }

    // Reactions and the row id come from the same authoritative rows as the
    // text, but a live/optimistic row that hasn't round-tripped yet carries
    // neither. Carry the cached copy forward so a reaction doesn't blink off
    // mid-turn. NEW object every time — the runtime repository's WeakMap
    // caches normalized ThreadMessages by ChatMessage identity.
    if (sameTurn && preserved.rowId === undefined && previous.rowId !== undefined) {
      preserved = { ...preserved, rowId: previous.rowId }
    }

    if (sameTurn && preserved.reactions === undefined && previous.reactions?.length) {
      preserved = { ...preserved, reactions: [...previous.reactions] }
    }

    const previousImages = embeddedImageUrls(previousText)

    if (!previousImages.length || embeddedImageUrls(chatMessageText(preserved)).length) {
      return preserved
    }

    if (nextText !== previousVisibleText) {
      return preserved
    }

    return withAppendedText(preserved, previousImages.map(url => `\n${url}`).join(''))
  })
}

/**
 * Keep the local tail of a turn while a reconnect hydrates an older server
 * projection. The user's optimistic row exists before prompt.submit persists
 * it, and the pending assistant row exists before message.complete commits it;
 * dropping either makes an accepted turn appear to vanish during transport
 * churn.
 *
 * A lagging projection can be behind by one live turn, never a whole local
 * history window. Preserve only the newest optimistic user row: compression
 * rewrites past context, so older `user-*` rows in a warm cache are stale
 * history, not in-flight work. The latest authoritative user confirms whether
 * that tail has persisted. An authoritative assistant at the same ordinal
 * supersedes the local stream only when it is at least as complete; an empty
 * or lagging inflight shell must not discard a fuller local pending reply
 * (#75825).
 *
 * Gateway bookkeeping markers (the model-switch / personality notices written
 * by tui_gateway/server.py) are persisted as role=user but are not user turns.
 * They must not take part in ordinal pairing on either side: a stored marker
 * between two real user turns shifts every later user ordinal, so the optimistic
 * row misses its committed copy and is appended a second time at the end of the
 * transcript — the duplicated user bubble of #67603.
 */
const isGatewaySystemMarker = (message: ChatMessage): boolean =>
  message.role === 'user' && chatMessageText(message).trimStart().startsWith('[System:')

/**
 * Does the row carry anything a viewer would miss — streamed answer text, or
 * the reasoning / tool-call structure the gateway's flat dump cannot express?
 * An empty inflight shell carries none of it.
 */
const hasStreamedContent = (message: ChatMessage): boolean =>
  chatMessageText(message).trim().length > 0 || hasStructuralParts(message)

/**
 * May the cached local row stand in for this authoritative assistant?
 *
 * Only for a live projection of the SAME reply that the local copy is further
 * along on: an empty shell, or text the local row strictly extends. Comparing
 * lengths alone lets an unrelated (merely longer) local row hijack the ordinal
 * — or the stream id — of a genuine stored reply. A retained failure snapshot
 * (`inflight.error`, projected with empty text) is never a shell: repainting it
 * from the local partial would hide the error and mark the turn healthy again.
 */
const localPendingSupersedes = (local: ChatMessage, authoritative: ChatMessage): boolean => {
  if (local.role !== 'assistant' || !isLiveTailRow(local)) {
    return false
  }

  if (!isLiveTailRow(authoritative) || authoritative.error) {
    return false
  }

  const authoritativeText = chatMessageText(authoritative).trim()

  if (!authoritativeText.length) {
    return hasStreamedContent(local)
  }

  const localText = chatMessageText(local).trim()

  return localText.length > authoritativeText.length && isStrictAnswerTextExtension(localText, authoritativeText)
}

/**
 * Take the cached row's content, but never its liveness. The renderer holds the
 * only copy of the streamed parts; the gateway remains the authority on whether
 * the turn is still running and on durable row identity — so a settled shell
 * must not repaint the reply as perpetually streaming.
 */
const withAuthoritativeTurnState = (local: ChatMessage, authoritative: ChatMessage): ChatMessage => {
  const merged: ChatMessage = { ...local, pending: authoritative.pending === true }

  if (local.rowId === undefined && authoritative.rowId !== undefined) {
    merged.rowId = authoritative.rowId
  }

  if (local.reactions === undefined && authoritative.reactions?.length) {
    merged.reactions = [...authoritative.reactions]
  }

  return merged
}

export function preserveLocalPendingTurnMessages(
  nextMessages: ChatMessage[],
  previousMessages: ChatMessage[]
): ChatMessage[] {
  if (!previousMessages.length) {
    return nextMessages
  }

  const nextByRoleOrdinal = new Map<string, ChatMessage>()
  const nextRoleCounts = new Map<ChatMessage['role'], number>()

  for (const message of nextMessages) {
    if (isGatewaySystemMarker(message)) {
      continue
    }

    const ordinal = nextRoleCounts.get(message.role) ?? 0
    nextRoleCounts.set(message.role, ordinal + 1)
    nextByRoleOrdinal.set(`${message.role}:${ordinal}`, message)
  }

  const nextIds = new Set(nextMessages.map(message => message.id))
  const previousRoleCounts = new Map<ChatMessage['role'], number>()

  const newestOptimisticUser = [...previousMessages]
    .reverse()
    .find(message => message.role === 'user' && message.id.startsWith('user-'))

  // A mid-turn redirect inserts its correction as a second optimistic user row
  // directly before the live reply, so one turn can own a contiguous RUN of
  // them. Preserving only the newest keeps the correction and drops the prompt
  // that started the turn. Widen to the run — but only the contiguous one: any
  // `user-*` row separated by an assistant reply is stale post-compression
  // history, which is what the newest-only rule exists to discard.
  const liveOptimisticUsers = new Set<ChatMessage>()

  if (newestOptimisticUser) {
    for (let index = previousMessages.indexOf(newestOptimisticUser); index >= 0; index -= 1) {
      const candidate = previousMessages[index]

      if (candidate.role !== 'user' || !candidate.id.startsWith('user-')) {
        break
      }

      liveOptimisticUsers.add(candidate)
    }
  }

  const latestAuthoritativeUser = [...nextMessages].reverse().find(message => message.role === 'user')
  const preserved: ChatMessage[] = []
  // Authoritative id → richer local pending row. Replacing (not appending)
  // avoids painting both the empty inflight shell and the full stream bubble.
  const replacements = new Map<string, ChatMessage>()

  for (const message of previousMessages) {
    if (isGatewaySystemMarker(message)) {
      continue
    }

    const ordinal = previousRoleCounts.get(message.role) ?? 0
    previousRoleCounts.set(message.role, ordinal + 1)

    const isOptimisticUser = message.role === 'user' && message.id.startsWith('user-')

    const isPendingAssistant =
      message.role === 'assistant' && (message.pending === true || message.id.startsWith('assistant-stream-'))

    if (!isOptimisticUser && !isPendingAssistant) {
      continue
    }

    // Same id already present: still prefer a strictly more complete local
    // pending body over an empty/stale shell that reused the stream id.
    if (nextIds.has(message.id)) {
      if (isPendingAssistant) {
        const existing = nextMessages.find(candidate => candidate.id === message.id)

        if (existing && localPendingSupersedes(message, existing)) {
          replacements.set(message.id, withAuthoritativeTurnState(message, existing))
        }
      }

      continue
    }

    if (isOptimisticUser && !liveOptimisticUsers.has(message)) {
      continue
    }

    if (
      isOptimisticUser &&
      latestAuthoritativeUser &&
      textWithoutReferenceLines(chatMessageText(latestAuthoritativeUser)) ===
        textWithoutReferenceLines(chatMessageText(message))
    ) {
      continue
    }

    const authoritative = nextByRoleOrdinal.get(`${message.role}:${ordinal}`)

    // A settled stream row (`pending: false` after message.complete) whose reply
    // the authoritative transcript already carries under its committed id is
    // stale: ordinal pairing can't see it, because the commit shifted the row
    // one ordinal earlier, and re-appending it renders the same answer twice
    // (#70209). Only text-identical rows are dropped — a settled row the backend
    // has NOT committed yet is the only copy of that reply and must survive.
    if (
      isPendingAssistant &&
      message.pending !== true &&
      nextMessages.some(
        candidate =>
          candidate.role === 'assistant' &&
          textWithoutReferenceLines(chatMessageText(candidate)) === textWithoutReferenceLines(chatMessageText(message))
      )
    ) {
      continue
    }

    if (authoritative) {
      if (isPendingAssistant) {
        // Keep the local pending row when it is the same reply further along
        // and the authoritative row is an empty projection shell or a prefix.
        // #75825
        if (!localPendingSupersedes(message, authoritative)) {
          continue
        }

        replacements.set(authoritative.id, withAuthoritativeTurnState(message, authoritative))

        continue
      }

      if (
        textWithoutReferenceLines(chatMessageText(authoritative)) ===
        textWithoutReferenceLines(chatMessageText(message))
      ) {
        continue
      }
    }

    // Ordinal pairing missed (the committed row shifted ordinal when history
    // was compacted / the authoritative list is shorter), yet the
    // authoritative transcript already carries this same reply under its
    // committed id. The #70209 guard above only covers SETTLED local rows
    // (`pending !== true`); a still-pending stream row that slips past
    // pairing falls through to `preserved.push` and renders the answer
    // twice — the reported A B C D E C D tail duplication.
    //
    // Three-way same-turn check against SETTLED authoritative rows only
    // (a live projection shell must not swallow the richer local row, see
    // the traces-only replacement test):
    //  1. identical answer text            -> authoritative already has it
    //  2. authoritative extends local text -> authoritative is the settled
    //     final version of the still-streaming local copy
    //  3. local extends authoritative text -> local is further along; replace
    //     the committed row with the richer body instead of appending
    if (isPendingAssistant) {
      const nextText = textWithoutReferenceLines(chatMessageText(message))

      const committedMatch = nextMessages.find(
        candidate =>
          candidate.role === 'assistant' &&
          !isLiveTailRow(candidate) &&
          (textWithoutReferenceLines(chatMessageText(candidate)) === nextText ||
            isStrictAnswerTextExtension(
              textWithoutReferenceLines(chatMessageText(candidate)),
              nextText
            ))
      )

      if (committedMatch) {
        continue
      }

      const committedPrefix = nextMessages.find(
        candidate =>
          candidate.role === 'assistant' &&
          !isLiveTailRow(candidate) &&
          isStrictAnswerTextExtension(
            nextText,
            textWithoutReferenceLines(chatMessageText(candidate))
          )
      )

      if (committedPrefix) {
        // Keep the COMMITTED id (not the local stream id): the turn is
        // already in the authoritative transcript, so the merged row must
        // stay addressable as that durable row — a stream id would read as a
        // live row again next reconcile and re-enter this same path.
        replacements.set(committedPrefix.id, {
          ...withAuthoritativeTurnState(message, committedPrefix),
          id: committedPrefix.id
        })

        continue
      }
    }

    preserved.push(message)
  }

  const withReplacements =
    replacements.size > 0 ? nextMessages.map(message => replacements.get(message.id) ?? message) : nextMessages

  return preserved.length ? [...withReplacements, ...preserved] : withReplacements
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
export function appendLiveSessionProjection(
  messages: ChatMessage[],
  projection: Pick<SessionResumeResponse, 'inflight' | 'queued' | 'session_id'>
): ChatMessage[] {
  const inflightUser = projection.inflight?.user?.trim() ?? ''
  const inflightAssistant = projection.inflight?.assistant ?? ''
  const inflightStreaming = Boolean(projection.inflight?.streaming)

  // Mid-turn redirect corrections. They are additional user bubbles belonging
  // to this same turn, ordered after the prompt that started it.
  const inflightCorrections = (projection.inflight?.corrections ?? [])
    .map(correction => correction?.trim() ?? '')
    .filter(Boolean)

  // A retained failed turn (the gateway keeps error snapshots replayable when
  // the terminal frame may have been lost to a disconnect) — surface the
  // failure on the projected row instead of rendering the partial as healthy.
  const inflightError = projection.inflight?.error?.trim() ?? ''
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
  // corrections), so match the contiguous run ending at the latest user row
  // rather than the single last one.
  const latestUserIndex = messages.map(message => message.role).lastIndexOf('user')
  const latestUserRun: ChatMessage[] = []

  for (let index = latestUserIndex; index >= 0 && messages[index].role === 'user'; index -= 1) {
    latestUserRun.unshift(messages[index])
  }

  const persistedInLatestRun = (text: string): boolean =>
    latestUserRun.some(
      message => textWithoutReferenceLines(chatMessageText(message)) === textWithoutReferenceLines(text)
    )

  const inflightUserAlreadyPersisted = Boolean(inflightUser) && persistedInLatestRun(inflightUser)

  if (inflightUser && !inflightUserAlreadyPersisted) {
    projected.push({
      id: `user-inflight-${sessionId}`,
      role: 'user',
      parts: [textPart(inflightUser)]
    })
  }

  // Corrections typed while the turn ran. Each is its own bubble, placed after
  // the original prompt and before the reply they redirected — the same order
  // the live transcript showed. Skip any the transcript already holds so a
  // resume doesn't double them.
  for (const [index, correction] of inflightCorrections.entries()) {
    if (persistedInLatestRun(correction)) {
      continue
    }

    projected.push({
      id: `user-inflight-correction-${index}-${sessionId}`,
      role: 'user',
      parts: [textPart(correction)]
    })
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

  if (inflightAssistant || inflightStreaming || inflightError || (inflightUser && queuedUser)) {
    if (turnAlreadyStructured && !inflightError) {
      // Structure is authoritative; skip the text-only dump row.
    } else {
      projected.push({
        id: liveStreamId,
        role: 'assistant',
        parts: inflightAssistant ? [assistantTextPart(inflightAssistant)] : [],
        pending: inflightStreaming,
        ...(inflightError ? { error: inflightError } : {})
      })
    }
  }

  if (queuedUser) {
    projected.push({
      id: `user-queued-${sessionId}`,
      role: 'user',
      parts: [textPart(queuedUser)]
    })
  }

  return projected.length ? [...messages, ...projected] : messages
}

export interface BranchMessage {
  content: string
  role: ChatMessage['role']
  source: ChatMessage
}

// The copyable spine of a branch: user/assistant turns that carry text.
export const toBranchMessages = (messages: ChatMessage[]): BranchMessage[] =>
  messages
    .map(message => ({ content: chatMessageText(message), role: message.role, source: message }))
    .filter(({ content, role }) => content.trim() && (role === 'assistant' || role === 'user'))

export function upsertOptimisticSession(
  created: SessionCreateResponse,
  id: string,
  title: string | null = null,
  preview: string | null = null,
  parentSessionId: string | null = null,
  lastActive?: number
) {
  const now = lastActive ?? Date.now() / 1000
  // Stamp the profile the session was just created on (= the live gateway's
  // profile) so the scoped sidebar shows the new row immediately instead of
  // filtering it out as "default" until the aggregator re-fetches.
  const profileKey = normalizeProfileKey($activeGatewayProfile.get())

  const session: SessionInfo = {
    // Seed cwd so the grouped sidebar can place the new row in its repo/worktree
    // lane immediately (the overlay groups by path); fall back to the workspace
    // the session was just started in when the create response omits it.
    cwd: created.info?.cwd ?? ($currentCwd.get().trim() || null),
    ended_at: null,
    id,
    input_tokens: 0,
    is_active: true,
    is_default_profile: profileKey === 'default',
    last_active: now,
    message_count: created.message_count ?? created.messages?.length ?? 0,
    model: created.info?.model ?? null,
    output_tokens: 0,
    parent_session_id: parentSessionId,
    preview,
    profile: profileKey,
    source: 'tui',
    started_at: now,
    title,
    tool_call_count: 0
  }

  setSessions(prev => [session, ...prev.filter(s => s.id !== id)])
}

export function patchSessionWorkspace(sessionId: string, cwd: string | undefined) {
  if (!cwd) {
    return
  }

  setSessions(prev => prev.map(session => (session.id === sessionId ? { ...session, cwd } : session)))
}

export function sessionShouldHaveTranscript(session: SessionInfo | undefined): boolean {
  return (session?.message_count ?? 0) > 0
}

function upsertResolvedSession(session: SessionInfo, storedSessionId: string) {
  const lineage = session._lineage_root_id ?? session.id

  setSessions(prev => [
    session,
    ...prev.filter(existing => {
      if (sessionMatchesStoredId(existing, storedSessionId)) {
        return false
      }

      return (existing._lineage_root_id ?? existing.id) !== lineage
    })
  ])
}

export async function resolveStoredSession(storedSessionId: string): Promise<SessionInfo | undefined> {
  const cached = $sessions.get().find(session => sessionMatchesStoredId(session, storedSessionId))

  // A row with no owning profile can't route a resume when more than one
  // profile exists — a resume without a profile lands on whichever gateway is
  // active (#67603 family, cross-profile open asymmetry). Treat such a hit as
  // unresolved and fall through to the by-id lookups, which stamp ownership.
  const multiProfile = $profiles.get().length > 1

  if (cached && (cached.profile?.trim() || !multiProfile)) {
    return cached
  }

  // Direct by-id on the live backend — one row lookup, no list scan. Covers
  // single-profile users and any id on the active profile (e.g. an old session
  // past the sidebar's recent window). 404 just means it's not on this profile.
  try {
    const session = await getSession(storedSessionId)

    // Older backends omit `profile` on unscoped GETs; the serving backend is
    // the active gateway's, so back-fill that rather than caching an unowned
    // row. A present stamp is preserved: in app-global remote mode a bare hit
    // can legitimately carry another profile's row (see the branch tests).
    session.profile ||= normalizeProfileKey($activeGatewayProfile.get())

    upsertResolvedSession(session, storedSessionId)

    return session
  } catch {
    // Not on the active profile — fall through to the cross-profile probe.
  }

  // Multi-profile only: probe each other profile by id (still one cheap lookup
  // each) rather than pulling every profile's recent sessions. The first hit
  // carries its owning `profile`, which routes the resume to the right backend.
  const activeKey = normalizeProfileKey($activeGatewayProfile.get())

  const otherProfiles = $profiles
    .get()
    .map(profile => normalizeProfileKey(profile.name))
    .filter(key => key !== activeKey)

  for (const profile of otherProfiles) {
    try {
      const session = await getSession(storedSessionId, profile)

      // Same ownership contract: the DESKTOP profile we explicitly probed is
      // authoritative, whatever the scoped backend stamped (older backends
      // omit the field; a per-profile remote override strips the alias before
      // forwarding, so that backend answers as its own "default").
      session.profile = profile

      upsertResolvedSession(session, storedSessionId)

      return session
    } catch {
      // Not on this profile; try the next.
    }
  }

  return undefined
}

/**
 * The profile that owns a stored session, resolved through the same
 * cache → active-backend → cross-profile ladder as `resolveStoredSession`.
 *
 * Recovery `session.resume` calls (stale runtime id, session-not-found, wedged
 * loop) must re-register the conversation on ITS backend, not on whichever
 * profile happens to be live. Omitting the profile lets the gateway fall back to
 * the launch-profile DB (tui_gateway/server.py), which is how a session bleeds
 * from one profile into another (#67603, second symptom). A cache-only lookup
 * misses any session outside the paginated sidebar window, so route through the
 * resolver, which probes uncached ids across profiles.
 */
export async function resolveSessionProfile(storedSessionId: null | string): Promise<string | undefined> {
  if (!storedSessionId) {
    return undefined
  }

  const profile = (await resolveStoredSession(storedSessionId))?.profile?.trim()

  return profile || undefined
}

type SessionRuntimeStatePatch = Partial<
  Pick<
    ClientSessionState,
    'branch' | 'cwd' | 'fast' | 'model' | 'personality' | 'provider' | 'reasoningEffort' | 'serviceTier' | 'yolo'
  >
>

interface ApplyRuntimeInfoOptions {
  /**
   * Whether this runtime belongs to the session the MAIN pane is showing.
   * Foreground (the default) mirrors into the composer atoms every main-pane
   * surface reads.
   *
   * A tile or a background branch must pass `false`: it owns a different
   * worktree, and writing its cwd into `$currentCwd` re-pointed the main
   * composer's coding rail (and the persisted workspace cwd) at the tile's
   * repo — the main rail painted a branch from a tree its session was never
   * in. The returned patch still carries every field, so the caller's own
   * per-session state is unaffected.
   */
  foreground?: boolean
}

/** Mirror a session's runtime state into the composer atoms the MAIN pane
 *  renders from. Foreground sessions only — see ApplyRuntimeInfoOptions. */
function publishRuntimeToComposer(state: SessionRuntimeStatePatch): void {
  if (state.model !== undefined) {
    setCurrentModel(state.model)
  }

  if (state.provider !== undefined) {
    setCurrentProvider(state.provider)
  }

  if (state.cwd !== undefined) {
    if (state.cwd) {
      // The runtime named a real folder for the session in the main pane, so
      // that conversation owns the path.
      commitWorkspaceCwdForSelectedSession(state.cwd)
    } else {
      // A detached session: the path on screen is provably still the previous
      // conversation's. Release rather than write `''` — clearing it collapses
      // the workspace/review panes on every switch.
      releaseWorkspaceCwdOwner()
    }
  }

  if (state.branch !== undefined) {
    setCurrentBranch(state.branch)
  }

  if (state.personality !== undefined) {
    setCurrentPersonality(state.personality)
  }

  if (state.reasoningEffort !== undefined) {
    setCurrentReasoningEffort(state.reasoningEffort)
  }

  if (state.serviceTier !== undefined) {
    setCurrentServiceTier(state.serviceTier)
  }

  if (state.fast !== undefined) {
    setCurrentFastMode(state.fast)
  }

  if (state.yolo !== undefined) {
    setYoloActive(state.yolo)
  }
}

export function applyRuntimeInfo(
  info: SessionRuntimeInfo | undefined,
  { foreground = true }: ApplyRuntimeInfoOptions = {}
): SessionRuntimeStatePatch | null {
  if (!info) {
    return null
  }

  // App/profile-level reporting is session-independent — a tile's runtime
  // reports backend skew and credential warnings just as usefully.
  reportBackendContract(info.desktop_contract)

  if (info.approval_mode !== undefined) {
    reconcileApprovalModeForProfile($activeGatewayProfile.get(), info.approval_mode)
  }

  requestDesktopOnboardingForCredentialWarning(info.credential_warning)

  reportInstallMethodWarning(info.install_warning)

  const sessionState: SessionRuntimeStatePatch = {}

  if (typeof info.model === 'string') {
    sessionState.model = info.model
  }

  if (typeof info.provider === 'string') {
    sessionState.provider = info.provider
  }

  // Empty string is authoritative, not "no opinion": a detached/bare session
  // reports `cwd: ''`, and the truthy-only test left `$currentCwd` — and so the
  // Files pane — pinned to the PREVIOUS project for the rest of the session
  // (#71254). Empty is routed through ownership release below rather than
  // persisted, so the pane hides a path it no longer owns instead of blanking.
  if (typeof info.cwd === 'string') {
    sessionState.cwd = info.cwd
  }

  if (info.branch !== undefined) {
    sessionState.branch = info.branch || ''
  }

  if (typeof info.personality === 'string') {
    sessionState.personality = normalizePersonalityValue(info.personality)
  }

  if (typeof info.reasoning_effort === 'string') {
    sessionState.reasoningEffort = info.reasoning_effort
  }

  if (typeof info.service_tier === 'string') {
    sessionState.serviceTier = info.service_tier
  }

  if (typeof info.fast === 'boolean') {
    sessionState.fast = info.fast
  }

  if (typeof info.yolo === 'boolean') {
    sessionState.yolo = info.yolo
  }

  if (foreground) {
    publishRuntimeToComposer(sessionState)

    if (info.usage) {
      setCurrentUsage(current => ({ ...current, ...info.usage }))
    }
  }

  return sessionState
}

export function applyStoredSessionPreviewRuntimeInfo(
  stored: { cwd?: null | string; model?: null | string } | undefined,
  storedSessionId: null | string
) {
  setCurrentModel(stored?.model || '')
  setCurrentProvider('')
  setCurrentReasoningEffort('')
  setCurrentServiceTier('')
  setCurrentFastMode(false)
  setYoloActive(false)
  setCurrentPersonality('')

  // Cold resume paints the transcript before `session.resume` returns, so
  // without this the Files pane shows the PREVIOUS project's tree for the whole
  // round-trip (#71254 / #76696). The sidebar row already knows this
  // conversation's workspace — `cwd` is part of the compact row projection — so
  // mirror it on the same tick the selection changes.
  //
  // Only `cwd` is consulted. `git_repo_root` is documented as null for non-git
  // workspaces and not-yet-backfilled history rows, so falling back to it would
  // read as "no workspace" for those sessions and blank a pane that was correct.
  const storedCwd = stored?.cwd?.trim() || ''

  if (storedCwd) {
    setCurrentCwdTransient(storedCwd)
    setWorkspaceCwdOwner(storedSessionId)
  } else {
    // Either a genuinely detached session, or a row outside the loaded sidebar
    // page (`stored` is undefined) — neither says anything about the workspace,
    // while `$currentCwd` still holds the previous conversation's folder.
    // Release so workspace-derived surfaces stop trusting it; `applyRuntimeInfo`
    // publishes the truth a moment later. The path is deliberately left in place
    // — clearing it collapses the workspace/review panes and drops file-tree
    // state on every switch.
    releaseWorkspaceCwdOwner()
  }

  // Same window, same reasoning: the branch is derived from the workspace, so
  // carrying the previous conversation's label across a switch is never right.
  setCurrentBranch('')
}

// A "session genuinely doesn't exist" failure (deleted, or an id from a wiped /
// rotated backend) — the REST transcript 404s with `Session not found`. Distinct
// from a transient/wedged backend (ECONNREFUSED, timeout), which must still
// retry rather than discard the id.
export function isSessionGoneError(err: unknown): boolean {
  const message = err instanceof Error ? err.message : String(err ?? '')

  return message.includes('404') || /session not found/i.test(message)
}
