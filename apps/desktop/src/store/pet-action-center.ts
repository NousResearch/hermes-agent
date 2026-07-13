import { atom, computed } from 'nanostores'

import type { ProfileInfo, SessionInfo } from '@/types/hermes'

import { $clarifyRequests, type ClarifyRequest } from './clarify'
import { $profiles, normalizeProfileKey } from './profile'
import type { StoredPromptRequest } from './prompt-identity'
import {
  $approvalRequests,
  $secretRequests,
  $sudoRequests,
  type ApprovalRequest
} from './prompts'
import { $cronSessions, $messagingSessions, $sessions } from './session'

export type PetActionCenterAllowedAction =
  | 'approve-always'
  | 'approve-once'
  | 'approve-session'
  | 'clarify-respond'
  | 'clarify-skip'
  | 'deny'
  | 'open-in-app'

interface PetActionCenterItemBase {
  actionable: boolean
  allowedActions: PetActionCenterAllowedAction[]
  blocking: true
  detail: string | null
  id: string
  profile: string
  profileLabel: string
  receivedAt: number
  sessionId: string | null
  sessionTitle: string | null
  storedSessionId: string | null
  summary: string | null
}

export interface PetActionCenterApprovalItem extends PetActionCenterItemBase {
  allowPermanent: boolean
  choices: string[] | null
  command: string
  description: string
  kind: 'approval'
  smartDenied: boolean
}

export interface PetActionCenterClarifyItem extends PetActionCenterItemBase {
  choices: string[] | null
  kind: 'clarify'
  question: string
}

export type PetActionCenterItem = PetActionCenterApprovalItem | PetActionCenterClarifyItem

export interface PetActionCenterState {
  action: PetActionCenterActionStatus | null
  actionableCount: number
  attentionCount: number
  blockingCount: number
  items: PetActionCenterItem[]
  secureInputCount: number
  selectedItemId: string | null
}

export type PetActionCenterErrorCode =
  | 'capability-denied'
  | 'disconnected'
  | 'invalid-item'
  | 'item-not-found'
  | 'open-failed'
  | 'rpc-failed'
  | 'session-unverified'

export type PetActionCenterActionStatus =
  | { status: 'submitting'; itemId: string }
  | { status: 'success'; itemId: string }
  | { status: 'stale'; itemId: string }
  | { status: 'error'; itemId: string; errorCode: PetActionCenterErrorCode }

interface PetActionCenterItemsState {
  actionableCount: number
  attentionCount: number
  blockingCount: number
  items: PetActionCenterItem[]
  secureInputCount: number
}

const $selectedPetActionCenterItemId = atom<string | null>(null)
const $petActionCenterActionStatus = atom<PetActionCenterActionStatus | null>(null)

export function setPetActionCenterActionStatus(status: PetActionCenterActionStatus): void {
  $petActionCenterActionStatus.set(status)
}

export function clearPetActionCenterActionStatus(): void {
  $petActionCenterActionStatus.set(null)
}

interface ProjectionContext {
  profiles: ProfileInfo[]
  sessions: SessionInfo[]
}

function profileLabel(profile: string, profiles: ProfileInfo[]): string {
  return profiles.find(candidate => normalizeProfileKey(candidate.name) === profile)?.name ?? profile
}

function sessionContext(
  request: StoredPromptRequest<ClarifyRequest | ApprovalRequest>,
  sessions: SessionInfo[]
): { sessionTitle: string | null; storedSessionId: string | null } {
  const profile = normalizeProfileKey(request.profile)
  const belongsToProfile = (session: SessionInfo) => normalizeProfileKey(session.profile) === profile

  const storedSessionId = request.storedSessionId

  const stored = storedSessionId
    ? sessions.find(
        session =>
          belongsToProfile(session) && (session.id === storedSessionId || session._lineage_root_id === storedSessionId)
      )
    : request.sessionId
      ? sessions.find(
          session =>
            belongsToProfile(session) &&
            (session.id === request.sessionId || session._lineage_root_id === request.sessionId)
        )
      : undefined

  return {
    sessionTitle: stored?.title?.trim() || null,
    storedSessionId: stored ? (storedSessionId ?? stored.id) : null
  }
}

function itemId(
  kind: PetActionCenterApprovalItem['kind'] | PetActionCenterClarifyItem['kind'],
  request: StoredPromptRequest<ClarifyRequest | ApprovalRequest>
): string {
  return JSON.stringify([kind, normalizeProfileKey(request.profile), request.sessionId ?? '', request.requestIdentity])
}

function commonItem(
  kind: PetActionCenterApprovalItem['kind'] | PetActionCenterClarifyItem['kind'],
  request: StoredPromptRequest<ClarifyRequest | ApprovalRequest>,
  context: ProjectionContext
): Pick<
  PetActionCenterItemBase,
  'blocking' | 'id' | 'profile' | 'profileLabel' | 'receivedAt' | 'sessionId' | 'sessionTitle' | 'storedSessionId'
> {
  const profile = normalizeProfileKey(request.profile)

  return {
    blocking: true,
    id: itemId(kind, request),
    profile,
    profileLabel: profileLabel(profile, context.profiles),
    receivedAt: request.receivedAt,
    sessionId: request.sessionId,
    ...sessionContext(request, context.sessions)
  }
}

function approvalActions(
  request: StoredPromptRequest<ApprovalRequest>,
  canOpenExactSession: boolean
): PetActionCenterAllowedAction[] {
  const choices = request.choices ?? (request.smartDenied ? ['once', 'deny'] : ['once', 'session', 'always', 'deny'])
  const actions: PetActionCenterAllowedAction[] = []

  if (choices.includes('once')) {
    actions.push('approve-once')
  }

  if (choices.includes('session')) {
    actions.push('approve-session')
  }

  if (request.allowPermanent !== false && choices.includes('always')) {
    actions.push('approve-always')
  }

  if (choices.includes('deny')) {
    actions.push('deny')
  }

  if (canOpenExactSession) {
    actions.push('open-in-app')
  }

  return actions
}

function approvalItem(
  request: StoredPromptRequest<ApprovalRequest>,
  context: ProjectionContext
): PetActionCenterApprovalItem {
  const description = request.description.trim()
  const command = request.command.trim()
  const common = commonItem('approval', request, context)

  return {
    ...common,
    actionable: true,
    allowPermanent: request.allowPermanent !== false,
    allowedActions: approvalActions(request, Boolean(common.storedSessionId)),
    choices: request.choices ? [...request.choices] : null,
    command,
    description,
    detail: command || null,
    kind: 'approval',
    smartDenied: request.smartDenied === true,
    summary: description || command || null
  }
}

function clarifyItem(
  request: StoredPromptRequest<ClarifyRequest>,
  context: ProjectionContext
): PetActionCenterClarifyItem {
  const choices = request.choices ? [...request.choices] : null
  const question = request.question.trim()
  const common = commonItem('clarify', request, context)

  return {
    ...common,
    actionable: true,
    allowedActions: [
      'clarify-respond',
      'clarify-skip',
      ...(common.storedSessionId ? (['open-in-app'] as const) : [])
    ],
    choices,
    detail: choices?.join('\n') || null,
    kind: 'clarify',
    question,
    summary: question || null
  }
}

function compareItems(left: PetActionCenterItem, right: PetActionCenterItem): number {
  if (left.actionable !== right.actionable) {
    return left.actionable ? -1 : 1
  }

  if (left.blocking !== right.blocking) {
    return left.blocking ? -1 : 1
  }

  return left.receivedAt - right.receivedAt || left.id.localeCompare(right.id)
}

const $petActionCenterItems = computed(
  [
    $approvalRequests,
    $clarifyRequests,
    $sudoRequests,
    $secretRequests,
    $profiles,
    $sessions,
    $cronSessions,
    $messagingSessions
  ],
  (
    approvals,
    clarifications,
    sudos,
    secrets,
    profiles,
    sessions,
    cronSessions,
    messagingSessions
  ): PetActionCenterItemsState => {
    const context = { profiles, sessions: [...sessions, ...cronSessions, ...messagingSessions] }
    const secureInputCount = Object.keys(sudos).length + Object.keys(secrets).length

    const items: PetActionCenterItem[] = [
      ...Object.values(approvals).map(request => approvalItem(request, context)),
      ...Object.values(clarifications).map(request => clarifyItem(request, context))
    ].sort(compareItems)

    return {
      actionableCount: items.filter(item => item.actionable).length,
      attentionCount: items.length + secureInputCount,
      blockingCount: items.filter(item => item.blocking).length + secureInputCount,
      items,
      secureInputCount
    }
  }
)

// Keep selection canonical rather than retaining a vanished id that could
// unexpectedly become selected again if a later request reused it. The item
// projection is already deterministically sorted, so index zero is the single
// fallback policy for initial selection and removals.
$petActionCenterItems.subscribe(state => {
  const selected = $selectedPetActionCenterItemId.get()
  const next = selected && state.items.some(item => item.id === selected) ? selected : (state.items[0]?.id ?? null)

  if (selected !== next) {
    $selectedPetActionCenterItemId.set(next)
  }
})

export function selectPetActionCenterItem(itemId: string): boolean {
  if (!$petActionCenterItems.get().items.some(item => item.id === itemId)) {
    return false
  }

  $selectedPetActionCenterItemId.set(itemId)

  return true
}

export const $petActionCenter = computed(
  [$petActionCenterItems, $selectedPetActionCenterItemId, $petActionCenterActionStatus],
  (state, selectedItemId, action): PetActionCenterState => ({ ...state, action, selectedItemId })
)
