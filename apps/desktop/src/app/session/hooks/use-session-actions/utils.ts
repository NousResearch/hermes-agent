import { getSession } from '@/hermes'
import { assistantTextPart, type ChatMessage, chatMessageText, textPart } from '@/lib/chat-messages'
import { normalizePersonalityValue } from '@/lib/chat-runtime'
import { embeddedImageUrls, textWithoutEmbeddedImages } from '@/lib/embedded-images'
import { reconcileApprovalModeForProfile } from '@/store/approval-mode'
import { requestDesktopOnboarding } from '@/store/onboarding'
import { $activeGatewayProfile, $profiles, normalizeProfileKey } from '@/store/profile'
import {
  $currentCwd,
  $sessions,
  sessionMatchesStoredId,
  setCurrentBranch,
  setCurrentCwd,
  setCurrentFastMode,
  setCurrentModel,
  setCurrentPersonality,
  setCurrentProvider,
  setCurrentReasoningEffort,
  setCurrentServiceTier,
  setCurrentUsage,
  setSessions,
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

function preserveReasoningParts(message: ChatMessage, previous: ChatMessage): ChatMessage {
  if (message.parts.some(part => part.type === 'reasoning')) {
    return message
  }

  const reasoningParts = previous.parts.filter(part => part.type === 'reasoning')

  return reasoningParts.length ? { ...message, parts: [...reasoningParts, ...message.parts] } : message
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
//
// If your new field affects what the user sees in the transcript, add it to
// COMPARED. If it's metadata that shouldn't trigger a re-render, add it to
// IGNORED.
const _chatMessageFieldsExhaustive: {
  [K in Exclude<keyof ChatMessage, (typeof COMPARED_FIELDS)[number] | (typeof IGNORED_FIELDS)[number]>]: never
} = {}

const COMPARED_FIELDS = ['id', 'role', 'pending', 'error', 'hidden', 'branchGroupId', 'interim'] as const
const IGNORED_FIELDS = ['timestamp', 'attachmentRefs', 'parts'] as const

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
    (a.interim ?? false) !== (b.interim ?? false)
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
    let preserved = message

    if (nextText === previousVisibleText || nextText === previousText.trim()) {
      preserved = preserveReasoningParts(preserved, previous)
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
 * Authoritative rows use different ids, so match by role ordinal. A matching
 * user row is considered committed only when its visible text also matches;
 * any authoritative assistant at the same ordinal supersedes the local stream.
 */
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
    const ordinal = nextRoleCounts.get(message.role) ?? 0
    nextRoleCounts.set(message.role, ordinal + 1)
    nextByRoleOrdinal.set(`${message.role}:${ordinal}`, message)
  }

  const nextIds = new Set(nextMessages.map(message => message.id))
  const previousRoleCounts = new Map<ChatMessage['role'], number>()
  const preserved: ChatMessage[] = []

  for (const message of previousMessages) {
    const ordinal = previousRoleCounts.get(message.role) ?? 0
    previousRoleCounts.set(message.role, ordinal + 1)

    const isOptimisticUser = message.role === 'user' && message.id.startsWith('user-')

    const isPendingAssistant =
      message.role === 'assistant' && (message.pending === true || message.id.startsWith('assistant-stream-'))

    if ((!isOptimisticUser && !isPendingAssistant) || nextIds.has(message.id)) {
      continue
    }

    const authoritative = nextByRoleOrdinal.get(`${message.role}:${ordinal}`)

    if (authoritative) {
      if (isPendingAssistant) {
        continue
      }

      if (chatMessageText(authoritative).trim() === chatMessageText(message).trim()) {
        continue
      }
    }

    preserved.push(message)
  }

  return preserved.length ? [...nextMessages, ...preserved] : nextMessages
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
  const queuedUser = projection.queued?.user?.trim() ?? ''

  if (!inflightUser && !inflightAssistant && !inflightStreaming && !queuedUser) {
    return messages
  }

  const sessionId = projection.session_id || 'session'
  const projected: ChatMessage[] = []

  if (inflightUser) {
    projected.push({
      id: `user-inflight-${sessionId}`,
      role: 'user',
      parts: [textPart(inflightUser)]
    })
  }

  // Keep a pending assistant boundary even before the first delta when a
  // queued user turn follows it. This preserves the two distinct turns.
  if (inflightAssistant || inflightStreaming || (inflightUser && queuedUser)) {
    projected.push({
      id: `assistant-stream-${sessionId}`,
      role: 'assistant',
      parts: inflightAssistant ? [assistantTextPart(inflightAssistant)] : [],
      pending: inflightStreaming
    })
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

  if (cached) {
    return cached
  }

  // Direct by-id on the live backend — one row lookup, no list scan. Covers
  // single-profile users and any id on the active profile (e.g. an old session
  // past the sidebar's recent window). 404 just means it's not on this profile.
  try {
    const session = await getSession(storedSessionId)

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

      upsertResolvedSession(session, storedSessionId)

      return session
    } catch {
      // Not on this profile; try the next.
    }
  }

  return undefined
}

type SessionRuntimeStatePatch = Partial<
  Pick<
    ClientSessionState,
    'branch' | 'cwd' | 'fast' | 'model' | 'personality' | 'provider' | 'reasoningEffort' | 'serviceTier' | 'yolo'
  >
>

export function applyRuntimeInfo(info: SessionRuntimeInfo | undefined): SessionRuntimeStatePatch | null {
  if (!info) {
    return null
  }

  const sessionState: SessionRuntimeStatePatch = {}

  reportBackendContract(info.desktop_contract)

  if (info.approval_mode !== undefined) {
    reconcileApprovalModeForProfile($activeGatewayProfile.get(), info.approval_mode)
  }

  if (info.credential_warning) {
    requestDesktopOnboarding(info.credential_warning)
  }

  reportInstallMethodWarning(info.install_warning)

  if (typeof info.model === 'string') {
    setCurrentModel(info.model)
    sessionState.model = info.model
  }

  if (typeof info.provider === 'string') {
    setCurrentProvider(info.provider)
    sessionState.provider = info.provider
  }

  if (info.cwd) {
    setCurrentCwd(info.cwd)
    sessionState.cwd = info.cwd
  }

  if (info.branch !== undefined) {
    setCurrentBranch(info.branch || '')
    sessionState.branch = info.branch || ''
  }

  if (typeof info.personality === 'string') {
    const personality = normalizePersonalityValue(info.personality)
    setCurrentPersonality(personality)
    sessionState.personality = personality
  }

  if (typeof info.reasoning_effort === 'string') {
    setCurrentReasoningEffort(info.reasoning_effort)
    sessionState.reasoningEffort = info.reasoning_effort
  }

  if (typeof info.service_tier === 'string') {
    setCurrentServiceTier(info.service_tier)
    sessionState.serviceTier = info.service_tier
  }

  if (typeof info.fast === 'boolean') {
    setCurrentFastMode(info.fast)
    sessionState.fast = info.fast
  }

  if (typeof info.yolo === 'boolean') {
    setYoloActive(info.yolo)
    sessionState.yolo = info.yolo
  }

  if (info.usage) {
    setCurrentUsage(current => ({ ...current, ...info.usage }))
  }

  return sessionState
}

export function applyStoredSessionPreviewRuntimeInfo(stored: { model?: null | string } | undefined) {
  setCurrentModel(stored?.model || '')
  setCurrentProvider('')
  setCurrentReasoningEffort('')
  setCurrentServiceTier('')
  setCurrentFastMode(false)
  setYoloActive(false)
  setCurrentPersonality('')
}

// A "session genuinely doesn't exist" failure (deleted, or an id from a wiped /
// rotated backend) — the REST transcript 404s with `Session not found`. Distinct
// from a transient/wedged backend (ECONNREFUSED, timeout), which must still
// retry rather than discard the id.
export function isSessionGoneError(err: unknown): boolean {
  const message = err instanceof Error ? err.message : String(err ?? '')

  return message.includes('404') || /session not found/i.test(message)
}
