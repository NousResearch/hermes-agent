import { useStore } from '@nanostores/react'
import { computed } from 'nanostores'

import { jobRunInFlight } from '@/app/cron/run-control'
import type { ClientSessionState } from '@/app/types'
import type { CronJob, SessionInfo } from '@/hermes'
import { agentSidebarKey } from '@/store/agent-sidebar-preferences'
import { $clarifyRequests } from '@/store/clarify'
import { $cronJobs, getCronJobsSnapshotScope } from '@/store/cron'
import {
  $approvalRequests,
  $secretRequests,
  $sudoRequests,
  $vaultCodeRequests,
  $vaultSaveLoginRequests,
  $vaultUnlockRequests,
  type ApprovalRequest
} from '@/store/prompts'
import { $cronSessions, $messagingSessions, $unlistedSessionOwnerRows, sessionMatchesStoredId } from '@/store/session'
import { $sessionDotStateById, type SessionDotState, showsRunningArc } from '@/store/session-dot-state'
import { $sessionStates, knownOwnerForSession, runtimeSessionOwner } from '@/store/session-states'
import { $unreadFinishedMarkers } from '@/store/session-unread'
import { $unreadWriteGuard, UNREAD_WRITE_GUARD_MS } from '@/store/session-unread-remote'

import { type SimplifiedAgent, useSimplifiedOwner } from './use-simplified-agents'

export interface AgentSidebarAttention {
  kind: 'approval' | 'input' | 'failed' | 'running' | 'unread'
  since?: number
  session?: SessionInfo
}

type ClaimAttention = (owner: string, attention: AgentSidebarAttention) => void

interface SessionAttentionSources {
  connectionId: string | null
  rows: SessionInfo[]
  ownersById: Map<string, Set<string>>
  states: Record<string, ClientSessionState>
  dots: Readonly<Record<string, SessionDotState>>
  approvals: Record<string, ApprovalRequest>
  inputs: Map<string, number | undefined>
}

const priority: Record<AgentSidebarAttention['kind'], number> = {
  approval: 0,
  input: 1,
  failed: 2,
  running: 3,
  unread: 4
}

const $pendingInputs = computed(
  [$clarifyRequests, $sudoRequests, $secretRequests, $vaultUnlockRequests, $vaultSaveLoginRequests, $vaultCodeRequests],
  (clarify, ...requests) => {
    // Clarify records actual receipt time in seconds. Other prompt kinds do
    // not expose a timestamp; do not present their render time as wait duration.
    const inputs = new Map(
      Object.entries(clarify).map(
        ([id, request]) => [id, request.receivedAt === undefined ? undefined : request.receivedAt * 1000] as const
      )
    )

    for (const items of requests) {
      for (const id of Object.keys(items)) {
        if (id && !inputs.has(id)) {
          inputs.set(id, undefined)
        }
      }
    }

    return inputs
  }
)

function rowOwnerKey(row: SessionInfo, connectionId: string | null): string {
  return agentSidebarKey({ connectionId: row.connection_id || connectionId, profile: row.profile || 'default' })
}

function sessionAliases(row: SessionInfo): string[] {
  return [row.id, row._lineage_root_id, ...(row._lineage_ids ?? [])].filter((id): id is string => Boolean(id))
}

function noteOwner(ownersById: Map<string, Set<string>>, id: string, key: string): void {
  const owners = ownersById.get(id) ?? new Set<string>()
  owners.add(key)
  ownersById.set(id, owners)
}

function runtimeOwnerKey(id: string, storedId: string, sources: SessionAttentionSources): string | undefined {
  const eventOwner = runtimeSessionOwner(id)
  const candidates = sources.ownersById.get(storedId)
  const owner = eventOwner ?? (candidates && candidates.size > 1 ? undefined : knownOwnerForSession(id, sources.rows))

  if (owner && typeof owner === 'object') {
    return agentSidebarKey({ connectionId: owner.connectionId ?? null, profile: owner.targetProfile || owner.profile })
  }

  if (typeof owner === 'string' && (eventOwner || candidates?.size !== 1)) {
    return agentSidebarKey({ connectionId: null, profile: owner })
  }

  return candidates?.size === 1 ? [...candidates][0] : undefined
}

function failedTurnAttention(state: ClientSessionState | undefined): AgentSidebarAttention | undefined {
  if (!state || state.busy || state.turnLive) {
    return
  }

  const last = state.messages.findLast(
    message => !message.hidden && (message.role === 'assistant' || message.role === 'user')
  )

  if (last?.role !== 'assistant' || last.pending || !last.error) {
    return
  }

  const since = last.completedAt ?? last.timestamp

  return { kind: 'failed', ...(since === undefined ? {} : { since: since * 1000 }) }
}

function runtimeAttention(
  state: ClientSessionState | undefined,
  dot: SessionDotState | undefined,
  approval: boolean,
  input: boolean,
  inputSince: number | undefined
): AgentSidebarAttention | undefined {
  if (approval) {
    return { kind: 'approval' }
  }

  if (input || state?.needsInput || dot === 'needs-input') {
    return { kind: 'input', ...(inputSince === undefined ? {} : { since: inputSince }) }
  }

  const failed = failedTurnAttention(state)

  if (failed) {
    return failed
  }

  if (state?.busy || state?.turnLive || showsRunningArc(dot ?? 'idle')) {
    return { kind: 'running', ...(state?.turnStartedAt ? { since: state.turnStartedAt } : {}) }
  }

  if (dot === 'unread') {
    return { kind: 'unread' }
  }
}

function collectRuntimeAttention(sources: SessionAttentionSources, claim: ClaimAttention): void {
  const { states, approvals, inputs, dots, rows, connectionId, ownersById } = sources

  const ids = new Set([
    ...Object.keys(states),
    ...Object.keys(approvals),
    ...inputs.keys(),
    ...Object.keys(dots).filter(id => !ownersById.has(id))
  ])

  const owned: Array<{ id: string; storedId: string; key: string }> = []

  // Establish every runtime's owner before consulting bare-id dot signals.
  for (const id of ids) {
    if (!id) {
      continue
    }

    const storedId = states[id]?.storedSessionId || id
    const key = runtimeOwnerKey(id, storedId, sources)

    if (key) {
      noteOwner(ownersById, storedId, key)
      owned.push({ id, storedId, key })
    }
  }

  for (const { id, storedId, key } of owned) {
    const dot = ownersById.get(storedId)?.size === 1 ? dots[storedId] : undefined
    const attention = runtimeAttention(states[id], dot, Boolean(approvals[id]), inputs.has(id), inputs.get(id))

    if (attention) {
      const session = rows.find(row => rowOwnerKey(row, connectionId) === key && sessionMatchesStoredId(row, storedId))
      claim(key, { ...attention, session })
    }
  }
}

function rowAttention(
  row: SessionInfo,
  sources: SessionAttentionSources,
  unreadGuard: ReadonlyMap<string, { at: number; value: boolean }>
): AgentSidebarAttention | undefined {
  const ids = sessionAliases(row)
  // Bare-id dot signals cannot cross colliding owners; row-local server truth can.
  const unambiguous = ids.every(id => sources.ownersById.get(id)?.size === 1)
  const dots = unambiguous ? ids.map(id => sources.dots[id]) : []
  const guard = unambiguous ? unreadGuard.get(row.id) : undefined
  const unread = guard && Date.now() - guard.at < UNREAD_WRITE_GUARD_MS ? guard.value : row.unread

  if (dots.includes('needs-input')) {
    return { kind: 'input', session: row }
  }

  if (dots.some(showsRunningArc) || row.scheduler_owned === true) {
    return { kind: 'running', session: row }
  }

  if (dots.includes('unread') || unread) {
    return { kind: 'unread', session: row }
  }
}

function collectUnreadAttention(
  agents: SimplifiedAgent[],
  rows: SessionInfo[],
  markers: Record<string, string[]>,
  claim: ClaimAttention
): void {
  // Profile-only legacy markers are attributable only to a single machine.
  for (const [profile, ids] of Object.entries(markers)) {
    const owners = new Set(
      agents
        .filter(agent => (agent.targetProfile || agent.profile) === profile)
        .map(agent => agentSidebarKey({ connectionId: agent.connectionId, profile }))
    )

    if (ids.length && owners.size === 1 && ids.some(id => !rows.some(row => sessionMatchesStoredId(row, id)))) {
      claim([...owners][0], { kind: 'unread' })
    }
  }
}

function collectScheduleAttention(jobs: CronJob[], claim: ClaimAttention): void {
  // Untagged snapshots cannot establish ownership. Fleet roster has no job status.
  const scope = getCronJobsSnapshotScope()

  if (!scope) {
    return
  }

  const [snapshotConnection, snapshotProfile] = scope.split('\u0000')

  for (const job of jobs) {
    const profile = snapshotProfile === 'all' ? job.profile : snapshotProfile

    if (!profile) {
      continue
    }

    const owner = agentSidebarKey({ connectionId: snapshotConnection || null, profile })
    const execution = job.latest_execution

    // A new run supersedes the previous result, as it does for chat turns.
    if (jobRunInFlight(job)) {
      const since = Date.parse((job.live_execution ?? execution)?.claimed_at ?? '')
      claim(owner, { kind: 'running', ...(Number.isFinite(since) ? { since } : {}) })

      continue
    }

    const executionFailed = execution
      ? execution.status === 'failed'
      : job.last_status === 'error' || Boolean(job.last_error)

    const deliveryFailed =
      execution?.delivery_outcome == null
        ? job.last_status === 'delivery_failed' || Boolean(job.last_delivery_error)
        : execution.delivery_outcome === 'failed'

    if (executionFailed || deliveryFailed) {
      const since = job.last_run_at ? Date.parse(job.last_run_at) : NaN
      claim(owner, { kind: 'failed', ...(Number.isFinite(since) ? { since } : {}) })
    }
  }
}

/** Read existing event/snapshot stores only. Hiding never changes subscriptions or backend state. */
export function useAgentSidebarAttention(
  agents: SimplifiedAgent[],
  sessions: SessionInfo[]
): Map<string, AgentSidebarAttention> {
  const connectionId = useSimplifiedOwner()
  const cron = useStore($cronSessions)
  const messaging = useStore($messagingSessions)
  const unlisted = useStore($unlistedSessionOwnerRows)
  const states = useStore($sessionStates)
  const dots = useStore($sessionDotStateById)
  const approvals = useStore($approvalRequests)
  const inputs = useStore($pendingInputs)
  const jobs = useStore($cronJobs)
  const markers = useStore($unreadFinishedMarkers)
  const unreadGuard = useStore($unreadWriteGuard)
  const result = new Map<string, AgentSidebarAttention>()
  const agentsByOwner = new Map<string, SimplifiedAgent[]>()

  for (const agent of agents) {
    const key = agentSidebarKey({ connectionId: agent.connectionId, profile: agent.targetProfile || agent.profile })
    const entries = agentsByOwner.get(key) ?? []
    entries.push(agent)
    agentsByOwner.set(key, entries)
  }

  const rows = [...sessions, ...messaging, ...cron, ...unlisted]
  const ownersById = new Map<string, Set<string>>()

  for (const row of rows) {
    for (const id of sessionAliases(row)) {
      noteOwner(ownersById, id, rowOwnerKey(row, connectionId))
    }
  }

  const claim: ClaimAttention = (key, attention) => {
    for (const agent of agentsByOwner.get(key) ?? []) {
      const previous = result.get(agent.id)

      if (
        !previous ||
        priority[attention.kind] < priority[previous.kind] ||
        (attention.kind === previous.kind && !previous.session && attention.session)
      ) {
        result.set(agent.id, attention)
      }
    }
  }

  const sources = { connectionId, rows, ownersById, states, dots, approvals, inputs }
  collectRuntimeAttention(sources, claim)

  for (const row of rows) {
    const attention = rowAttention(row, sources, unreadGuard)

    if (attention) {
      claim(rowOwnerKey(row, connectionId), attention)
    }
  }

  collectUnreadAttention(agents, rows, markers, claim)
  collectScheduleAttention(jobs, claim)

  return result
}
