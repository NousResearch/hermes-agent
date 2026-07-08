import { getSession } from '@/hermes'
import { type ChatMessage, chatMessageText } from '@/lib/chat-messages'
import { normalizePersonalityValue } from '@/lib/chat-runtime'
import { embeddedImageUrls, textWithoutEmbeddedImages } from '@/lib/embedded-images'
import { requestDesktopOnboarding } from '@/store/onboarding'
import { $activeGatewayProfile, $profiles, normalizeProfileKey } from '@/store/profile'
import {
  $currentCwd,
  $sessions,
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
import { reportBackendContract, reportInstallMethodWarning } from '@/store/updates'
import type { SessionCreateResponse, SessionInfo, SessionRuntimeInfo } from '@/types/hermes'

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

function chatMessagesEquivalent(a: ChatMessage, b: ChatMessage): boolean {
  if (
    a.id !== b.id ||
    a.role !== b.role ||
    a.pending !== b.pending ||
    a.error !== b.error ||
    a.hidden !== b.hidden ||
    a.branchGroupId !== b.branchGroupId
  ) {
    return false
  }

  if (a.parts.length !== b.parts.length) {
    return false
  }

  return a.parts.every((part, index) => JSON.stringify(part) === JSON.stringify(b.parts[index]))
}

export function chatMessageArraysEquivalent(a: ChatMessage[], b: ChatMessage[]): boolean {
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

export function sessionMatchesStoredId(session: SessionInfo, storedSessionId: string, profile?: null | string): boolean {
  const idMatches = session.id === storedSessionId || session._lineage_root_id === storedSessionId

  if (!idMatches) {
    return false
  }

  return !profile || normalizeProfileKey(session.profile) === normalizeProfileKey(profile)
}

function sameSessionProfile(a?: null | string, b?: null | string): boolean {
  return normalizeProfileKey(a) === normalizeProfileKey(b)
}

export function sessionShouldHaveTranscript(session: SessionInfo | undefined): boolean {
  return (session?.message_count ?? 0) > 0
}

function upsertResolvedSession(session: SessionInfo, storedSessionId: string, profile?: null | string) {
  const lineage = session._lineage_root_id ?? session.id
  const sessionProfile = profile ?? session.profile

  setSessions(prev => [
    session,
    ...prev.filter(existing => {
      if (sessionMatchesStoredId(existing, storedSessionId, sessionProfile)) {
        return false
      }

      return (existing._lineage_root_id ?? existing.id) !== lineage || !sameSessionProfile(existing.profile, sessionProfile)
    })
  ])
}

export async function resolveStoredSession(
  storedSessionId: string,
  requestedProfile?: null | string
): Promise<SessionInfo | undefined> {
  const requestedProfileKey = requestedProfile?.trim() || null
  const cached = $sessions.get().find(session => sessionMatchesStoredId(session, storedSessionId, requestedProfileKey))

  if (cached) {
    return cached
  }

  // Direct by-id on the live backend — one row lookup, no list scan. Covers
  // single-profile users and any id on the active profile (e.g. an old session
  // past the sidebar's recent window). 404 just means it's not on this profile.
  try {
    const session = await getSession(storedSessionId, requestedProfileKey)

    upsertResolvedSession(session, storedSessionId, requestedProfileKey)

    return session
  } catch {
    // Not on the active profile — fall through to the cross-profile probe.
  }

  if (requestedProfileKey) {
    return undefined
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

      upsertResolvedSession(session, storedSessionId, profile)

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
