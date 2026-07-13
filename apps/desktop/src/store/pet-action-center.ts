import type { ConnectionState } from '@hermes/shared'
import { atom, computed } from 'nanostores'

import type { ProfileInfo, SessionInfo } from '@/types/hermes'

import { $clarifyRequests, type ClarifyRequest } from './clarify'
import { $queuedPromptsBySession, type QueueState } from './composer-queue'
import { $gatewayStatesByProfile } from './gateway'
import { $petLiveSessions, type PetLiveSessionSnapshot } from './pet-live-session'
import { $activeGatewayProfile, $profiles, normalizeProfileKey } from './profile'
import type { StoredPromptRequest } from './prompt-identity'
import {
  $approvalRequests,
  $secretRequests,
  $sudoRequests,
  type ApprovalRequest
} from './prompts'
import { $activeSessionId, $cronSessions, $messagingSessions, $sessions } from './session'
import { profileSessionKey } from './session-identity'

export type PetActionCenterAllowedAction =
  | 'approve-always'
  | 'approve-once'
  | 'approve-session'
  | 'clarify-respond'
  | 'clarify-skip'
  | 'deny'
  | 'open-in-app'
  | 'send'
  | 'steer'
  | 'queue'
  | 'stop'
  | 'acknowledge'

export type PetActionCenterLiveTurnStatus = 'idle' | 'working' | 'waiting' | 'reviewing' | 'done' | 'failed'

export interface PetActionCenterLiveStatus {
  activityKind: PetLiveSessionSnapshot['activityKind']
  activityName: string | null
  connectionState: ConnectionState
  queuedCount: number
  status: PetActionCenterLiveTurnStatus
  turnStartedAt: number | null
}

interface PetActionCenterItemBase {
  actionable: boolean
  allowedActions: PetActionCenterAllowedAction[]
  blocking: boolean
  detail: string | null
  id: string
  profile: string
  profileLabel: string
  receivedAt: number
  sessionId: string | null
  sessionTitle: string | null
  storedSessionId: string | null
  summary: string | null
  liveStatus?: PetActionCenterLiveStatus
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

export interface PetActionCenterLiveTurnItem extends PetActionCenterItemBase, PetActionCenterLiveStatus {
  kind: 'live-turn'
}

export type PetActionCenterItem = PetActionCenterApprovalItem | PetActionCenterClarifyItem | PetActionCenterLiveTurnItem

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
  | 'invalid-text'
  | 'stale-runtime'

export type PetActionCenterActionStatus =
  | { status: 'submitting'; itemId: string }
  | { status: 'success'; itemId: string }
  | { status: 'stale'; itemId: string }
  | { status: 'steer-rejected' | 'steered' | 'queued' | 'stopped' | 'acknowledged'; itemId: string }
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

function sessionForStored(profile: string, storedSessionId: string | null, sessions: SessionInfo[]): SessionInfo | null {
  if (!storedSessionId) {
    return null
  }

  return (
    sessions.find(
      session =>
        normalizeProfileKey(session.profile) === profile &&
        (session.id === storedSessionId || session._lineage_root_id === storedSessionId)
    ) ?? null
  )
}

function liveTurnStatus(snapshot: PetLiveSessionSnapshot): PetActionCenterLiveTurnStatus {
  if (snapshot.outcome === 'failed') {
    return 'failed'
  }

  if (snapshot.outcome === 'done') {
    return 'done'
  }

  if (snapshot.needsInput) {
    return 'waiting'
  }

  if (snapshot.busy && snapshot.activityKind === 'reasoning') {
    return 'reviewing'
  }

  if (snapshot.busy || snapshot.awaitingResponse) {
    return 'working'
  }

  return 'idle'
}

function liveStatus(
  snapshot: PetLiveSessionSnapshot,
  connectionStates: Record<string, ConnectionState>,
  queues: QueueState
): PetActionCenterLiveStatus {
  const profile = normalizeProfileKey(snapshot.profile)
  const queueId = snapshot.storedSessionId ?? snapshot.runtimeSessionId

  return {
    activityKind: snapshot.activityKind,
    activityName: snapshot.activityName,
    connectionState: connectionStates[profile] ?? 'closed',
    queuedCount: queues[profileSessionKey(profile, queueId)]?.length ?? 0,
    status: liveTurnStatus(snapshot),
    turnStartedAt: snapshot.turnStartedAt
  }
}

function liveTurnActions(
  status: PetActionCenterLiveTurnStatus,
  connectionState: ConnectionState,
  canOpenExactSession: boolean
): PetActionCenterAllowedAction[] {
  const actions: PetActionCenterAllowedAction[] = []

  if (status === 'done' || status === 'failed') {
    actions.push('acknowledge')
  } else if (status === 'idle') {
    if (connectionState === 'open') {
      actions.push('send')
    }
  } else if (status === 'working' || status === 'reviewing') {
    if (connectionState === 'open') {
      actions.push('steer')
    }

    actions.push('queue')

    if (connectionState === 'open') {
      actions.push('stop')
    }
  }

  if (canOpenExactSession) {
    actions.push('open-in-app')
  }

  return actions
}

function liveTurnItem(
  snapshot: PetLiveSessionSnapshot,
  context: ProjectionContext,
  connectionStates: Record<string, ConnectionState>,
  queues: QueueState
): PetActionCenterLiveTurnItem {
  const profile = normalizeProfileKey(snapshot.profile)
  const status = liveStatus(snapshot, connectionStates, queues)
  const stored = sessionForStored(profile, snapshot.storedSessionId, context.sessions)
  const allowedActions = liveTurnActions(status.status, status.connectionState, Boolean(stored))

  return {
    ...status,
    actionable: allowedActions.length > 0,
    allowedActions,
    blocking: status.status === 'waiting',
    detail: null,
    id: JSON.stringify(['live-turn', profile, snapshot.runtimeSessionId]),
    kind: 'live-turn',
    profile,
    profileLabel: profileLabel(profile, context.profiles),
    receivedAt: snapshot.updatedAt,
    sessionId: snapshot.runtimeSessionId,
    sessionTitle: stored?.title?.trim() || null,
    storedSessionId: snapshot.storedSessionId,
    summary: null
  }
}

function compareItems(left: PetActionCenterItem, right: PetActionCenterItem): number {
  const leftPrompt = left.kind !== 'live-turn'
  const rightPrompt = right.kind !== 'live-turn'

  if (leftPrompt !== rightPrompt) {
    return leftPrompt ? -1 : 1
  }

  if (!leftPrompt && left.actionable !== right.actionable) {
    return left.actionable ? -1 : 1
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
    $messagingSessions,
    $petLiveSessions,
    $gatewayStatesByProfile,
    $queuedPromptsBySession,
    $activeGatewayProfile,
    $activeSessionId
  ],
  (
    approvals,
    clarifications,
    sudos,
    secrets,
    profiles,
    sessions,
    cronSessions,
    messagingSessions,
    liveSnapshots,
    connectionStates,
    queues,
    activeProfile,
    activeSessionId
  ): PetActionCenterItemsState => {
    const context = { profiles, sessions: [...sessions, ...cronSessions, ...messagingSessions] }
    const secureInputCount = Object.keys(sudos).length + Object.keys(secrets).length

    const visiblePromptItems: PetActionCenterItem[] = [
      ...Object.values(approvals).map(request => approvalItem(request, context)),
      ...Object.values(clarifications).map(request => clarifyItem(request, context))
    ]

    const liveByRuntime = new Map(
      liveSnapshots.map(snapshot => [profileSessionKey(snapshot.profile, snapshot.runtimeSessionId), snapshot])
    )

    const visiblePromptKeys = new Set(
      visiblePromptItems.flatMap(item =>
        item.sessionId ? [profileSessionKey(item.profile, item.sessionId)] : []
      )
    )

    const securePromptKeys = new Set(
      [...Object.values(sudos), ...Object.values(secrets)].flatMap(request =>
        request.sessionId ? [profileSessionKey(request.profile, request.sessionId)] : []
      )
    )

    for (const item of visiblePromptItems) {
      if (!item.sessionId) {
        continue
      }

      const snapshot = liveByRuntime.get(profileSessionKey(item.profile, item.sessionId))

      if (snapshot) {
        item.liveStatus = liveStatus(snapshot, connectionStates, queues)
      }
    }

    const normalizedActiveProfile = normalizeProfileKey(activeProfile)

    const liveItems = liveSnapshots
      .filter(snapshot => {
        const key = profileSessionKey(snapshot.profile, snapshot.runtimeSessionId)

        if (visiblePromptKeys.has(key)) {
          return false
        }

        return (
          liveTurnStatus(snapshot) !== 'idle' ||
          (normalizeProfileKey(snapshot.profile) === normalizedActiveProfile && snapshot.runtimeSessionId === activeSessionId)
        )
      })
      .map(snapshot => liveTurnItem(snapshot, context, connectionStates, queues))

    const items: PetActionCenterItem[] = [...visiblePromptItems, ...liveItems].sort(compareItems)

    const uncoveredAttention = liveItems.filter(item => {
      if (item.status !== 'waiting' && item.status !== 'done' && item.status !== 'failed') {
        return false
      }

      return !securePromptKeys.has(profileSessionKey(item.profile, item.sessionId ?? ''))
    })

    return {
      actionableCount: items.filter(item => item.actionable).length,
      attentionCount: visiblePromptItems.length + secureInputCount + uncoveredAttention.length,
      blockingCount:
        visiblePromptItems.filter(item => item.blocking).length +
        secureInputCount +
        uncoveredAttention.filter(item => item.status === 'waiting').length,
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
