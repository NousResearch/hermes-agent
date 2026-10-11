import type { ToolCallMessagePart } from '@assistant-ui/react'
import type {
  CatalogApproved,
  CatalogAppState,
  CatalogScan,
  CatalogServerError,
  CatalogTier,
  ConnectionAnswer,
  ConnectionOperationStatus,
  ConnectionOperationTarget,
  ConnectionRequestPayload,
  ConnectionSettleReason,
  ConnectionTargetAction,
  ConnectionTargetEnvField,
  ConnectionTargetKind,
  ConnectionTargetState,
  ConnectionUpdatePayload,
  InstallPhase
} from '@hermes/shared'
import { atom, computed } from 'nanostores'

import { resolveSessionOwner } from '@/app/session/hooks/use-session-actions/utils'
import type { SetupField } from '@/components/ui/setup-field-list'
import { connectorText } from '@/lib/connector-tools'

import { $gateway, requestGatewayForAgent } from './gateway'
import { $activeGatewayProfile } from './profile'
import { assertSessionOwnerResolved } from './session-owner-resolution'
import { isSessionOwnerRoute } from './session-request-router'

/** The backend sends ``prompt`` as null when the catalog entry has none; the form takes an absent one. */
const envFields = (fields: ConnectionTargetEnvField[] | null | undefined): SetupField[] =>
  (fields ?? []).map(({ default: defaultValue, name, prompt, required, secret }) => ({
    default: defaultValue,
    name,
    prompt: prompt ?? undefined,
    required,
    secret
  }))

export type {
  ConnectionSettleReason,
  ConnectionTargetAction,
  ConnectionTargetEnvField,
  ConnectionTargetKind,
  ConnectionTargetState
}

/** What the catalog says about a `plugin` / `skill` row (CATALOG-ROW-CONTRACT.md). The host resolved all of
 *  it; the model supplied only the id. */
export interface CatalogEntry {
  display: string
  description: string
  tier: CatalogTier | null
  /** Empty when the entry runs everywhere; the card shows a platform only when it is restricted. */
  platforms: string[]
  repo: string | null
  sha: string | null
  subdir: string | null
  scan: CatalogScan | null
  /** The Hermes version range the plugin needs; its env vars are the target's `requiredEnv`. */
  requiresHermes: string | null
  hasDesktopHalf: boolean
  /** The profile the row installs into; null when the chat's home is no named profile. */
  targetProfile: string | null
  appState: CatalogAppState | null
  /** On an installed skill row: the qualified name the model can now load. */
  skill: string | null
  /** The install step while the row is installing; the card words it. */
  phase: InstallPhase | null
  /** The non-secret choices the user approved; a Try again after the card settled repeats them. */
  approved: CatalogApproved | null
  /** Facts on an installed row: whether it was turned on, what it still needs, what did not connect. */
  enabled: boolean | null
  missingEnv: string[]
  serverErrors: CatalogServerError[]
  alreadyInstalled: boolean
  /** The backend's word that the row needs nothing more from the user: installed, skipped or failed. */
  resolved: boolean
}

/** One target of the operation as the renderer knows it. State comes only from the backend
 *  (`connection.request`, `connectors.operation.status`, `connection.update`); the card never sets it. */
export interface ConnectionTarget {
  name: string
  kind: ConnectionTargetKind
  action: ConnectionTargetAction
  state: ConnectionTargetState
  detail: string
  connectUrl: null | string
  /** The vendor account of a managed target once a mint named one; empty before that and on MCP targets. */
  connectionId: string
  /** The account name a managed target connects under; absent when the account has none. */
  alias?: null | string
  /** Toolkit metadata on connector targets; empty on an MCP target. */
  tools: string[]
  /** Credentials an MCP install is still waiting for; empty on every other target. */
  requiredEnv: SetupField[]
  instructions: string | null
  discoveryError: string | null
  /** Present on `plugin` and `skill` rows only. */
  catalog?: CatalogEntry
}

/** The session's connection operation. `deadlineAt`, `opId`, `targets[].state`, `settled` and
 *  `settledBy` are backend-owned; the renderer holds a cache and drives it through `connection.respond`. */
export interface ConnectionOwner {
  connectionId: null | string
  profile: string
}

export interface ConnectionRequest {
  /** The model's tool call that opened the operation. Call ids repeat across turns, so the row names its
   *  card by op id (`CONNECTION_OP_ARG`); the call id is only the fallback for a row without one. */
  toolCallId: string
  opId: string
  /** The sequence of the newest frame this cache holds; an older frame for the same op is dropped. */
  seq: number
  /** Unix seconds; backend-owned. */
  deadlineAt: number
  targets: ConnectionTarget[]
  settled: boolean
  settledBy: ConnectionSettleReason | null
  /** Local receipt time (Unix seconds), used to reject stale resume cleanup. */
  receivedAt?: number
  sessionId: string | null
}

const keyFor = (sessionId: string | null | undefined): string => sessionId ?? ''

export const $connectionRequests = atom<Record<string, ConnectionRequest>>({})

/** Settled operations the session's current one replaced, oldest first. Their cards stay drawn on the
 *  rows that opened them: every settled install card, and the last settled connect card when the current
 *  operation is an install one. A later connect operation still replaces the connect card, so consecutive
 *  connect calls show one card. */
const $keptConnectionRequests = atom<Record<string, ConnectionRequest[]>>({})

/** The key a tool row's args carry its operation id under, written when `connection.request` arrives.
 *  Models reuse call ids (`call_0` on every turn), so the op id, not the call id, names a row's card. */
export const CONNECTION_OP_ARG = 'hermes_connection_op'

export const connectionOpOf = (args: ToolCallMessagePart['args']): null | string =>
  connectorText(args[CONNECTION_OP_ARG]) || null

/** The operation a tool row draws. A row that carries an op id draws that operation and no other. A row
 *  without one (history loaded from disk, or a call whose request has not arrived yet) falls back to its
 *  call id, but only when a single known operation has that id, and a running row never takes a settled
 *  one: that is an earlier call's card. */
export const toolConnectionRequest = (
  sessionId: string | null,
  toolCallId: string,
  opId: null | string,
  running: boolean
) =>
  computed([$connectionRequests, $keptConnectionRequests], (requests, kept) => {
    const key = keyFor(sessionId)

    const known = [requests[key], ...(kept[key] ?? [])].filter((request): request is ConnectionRequest =>
      Boolean(request)
    )

    if (opId) {
      return known.find(request => request.opId === opId) ?? null
    }

    const matches = known.filter(request => request.toolCallId === toolCallId)
    const only = matches.length === 1 ? matches[0] : null

    return only && !(running && only.settled) ? only : null
  })

const TARGET_STATES: readonly ConnectionTargetState[] = [
  'connected',
  'expired',
  'failed',
  'initiated',
  'not_connected',
  'pending',
  'skipped'
]

const KINDS: readonly ConnectionTargetKind[] = ['connector', 'mcp', 'plugin', 'skill']
const ACTIONS: readonly ConnectionTargetAction[] = ['authorize', 'connect', 'enable', 'install', 'reconnect']
const SETTLE_REASONS: readonly ConnectionSettleReason[] = ['all_resolved', 'continue', 'deadline', 'interrupt']

// The wire carries these as typed literals already; the lookups defend against a backend a version ahead.
const oneOf =
  <T extends string>(allowed: readonly T[]) =>
  (value: null | string | undefined): T | undefined =>
    allowed.find(candidate => candidate === value)

const targetKind = oneOf(KINDS)
const targetState = oneOf(TARGET_STATES)
const targetAction = oneOf(ACTIONS)
const settleReason = oneOf(SETTLE_REASONS)

export const isCatalogKind = (kind: ConnectionTargetKind): kind is 'plugin' | 'skill' =>
  kind === 'plugin' || kind === 'skill'

/** A `manage_catalog` install operation, as opposed to a `manage_connections` one. */
export const isCatalogRequest = (request: ConnectionRequest): boolean =>
  request.targets.some(target => isCatalogKind(target.kind))

type InstalledFacts = Pick<CatalogEntry, 'alreadyInstalled' | 'enabled' | 'missingEnv' | 'serverErrors'>

const installedFacts = (entry: ConnectionOperationTarget): InstalledFacts => ({
  alreadyInstalled: entry.already_installed ?? false,
  enabled: entry.enabled ?? null,
  missingEnv: entry.missing_env ?? [],
  serverErrors: entry.server_errors ?? []
})

function catalogEntry(entry: ConnectionOperationTarget, name: string): CatalogEntry {
  return {
    ...installedFacts(entry),
    appState: entry.app_state ?? null,
    approved: entry.approved ?? null,
    description: entry.description ?? '',
    display: entry.display?.trim() || name,
    hasDesktopHalf: entry.has_desktop_half ?? false,
    phase: entry.phase ?? null,
    platforms: entry.platforms ?? [],
    repo: entry.repo ?? null,
    requiresHermes: entry.requires_hermes ?? null,
    resolved: entry.resolved ?? false,
    scan: entry.scan ?? null,
    sha: entry.sha ?? null,
    skill: entry.skill ?? null,
    subdir: entry.subdir ?? null,
    targetProfile: entry.target_profile?.trim() || null,
    tier: entry.tier ?? null
  }
}

// Every frame carries a fresh object; a field-equal entry keeps the old reference so the row does not churn.
const sameCatalog = (next: CatalogEntry | undefined, previous: CatalogEntry | undefined): boolean =>
  next === previous || JSON.stringify(next) === JSON.stringify(previous)

export function parseConnectionTarget(entry: ConnectionOperationTarget): ConnectionTarget | null {
  const name = entry.name.trim()

  if (!name) {
    return null
  }

  // An unknown kind from a backend a version ahead renders as the generic MCP row.
  const kind = targetKind(entry.kind) ?? 'mcp'

  return {
    action: targetAction(entry.action) ?? 'install',
    catalog: isCatalogKind(kind) ? catalogEntry(entry, name) : undefined,
    connectUrl: entry.connect_url ?? null,
    detail: entry.detail ?? '',
    kind,
    name,
    state: targetState(entry.state) ?? 'pending',
    tools: entry.tools ?? [],
    connectionId: entry.connection_id ?? '',
    alias: entry.alias ?? null,
    requiredEnv: envFields(entry.required_env),
    instructions: entry.instructions ?? null,
    discoveryError: entry.discovery_error ?? null
  }
}

/** Parse a `connection.request` event or the `pending_connection` resume field. Null when the payload
 *  carries no usable operation (no op id, no deadline, no targets). */
export function normalizeConnectionRequest(
  payload: ConnectionRequestPayload | null | undefined,
  sessionId: string | null
): ConnectionRequest | null {
  if (!payload) {
    return null
  }

  const targets = payload.targets
    .map(parseConnectionTarget)
    .filter((target): target is ConnectionTarget => target !== null)

  if (!payload.op_id || !payload.tool_call_id || !(payload.deadline_at > 0) || targets.length === 0) {
    return null
  }

  return {
    deadlineAt: payload.deadline_at,
    opId: payload.op_id,
    receivedAt: Date.now() / 1000,
    seq: payload.seq,
    sessionId,
    settled: false,
    settledBy: null,
    targets,
    toolCallId: payload.tool_call_id
  }
}

/** Overlay the authoritative `connectors.operation.status` snapshot on the cached request. Frames for
 *  another operation, and frames the operation wrote before the one already applied, change nothing:
 *  the transport can reorder them and an older one would regress a row. */
export function applyOperationStatus(request: ConnectionRequest, status: ConnectionOperationStatus): ConnectionRequest {
  if (status.op_id !== request.opId || status.seq <= request.seq) {
    return request
  }

  const byName = new Map(status.targets.map(target => [target.name, target] as const))

  const targets = request.targets.map(target => {
    const live: ConnectionOperationTarget | undefined = byName.get(target.name)

    return live ? mergeLiveTarget(target, live) : target
  })

  const settledBy = settleReason(status.settled_by) ?? null

  // Same reference on a no-op so subscribers do not re-render for an identical frame.
  const unchanged =
    request.deadlineAt === status.deadline_at &&
    request.seq === status.seq &&
    request.settled === status.settled &&
    request.settledBy === settledBy &&
    targets.every((target, index) => target === request.targets[index])

  return unchanged
    ? request
    : { ...request, deadlineAt: status.deadline_at, seq: status.seq, settled: status.settled, settledBy, targets }
}

function mergeLiveTarget(target: ConnectionTarget, live: ConnectionOperationTarget): ConnectionTarget {
  const liveCatalog = target.catalog ? catalogEntry(live, target.name) : undefined

  const next: ConnectionTarget = {
    ...target,
    catalog: sameCatalog(liveCatalog, target.catalog) ? target.catalog : liveCatalog,
    connectUrl: live.connect_url ?? target.connectUrl,
    detail: live.detail ?? target.detail,
    state: live.state,
    tools: live.tools ?? target.tools,
    connectionId: live.connection_id ?? target.connectionId,
    alias: live.alias === undefined ? target.alias : live.alias,
    requiredEnv: live.required_env ? envFields(live.required_env) : target.requiredEnv,
    instructions: live.instructions === undefined ? target.instructions : live.instructions,
    discoveryError: live.discovery_error === undefined ? target.discoveryError : live.discovery_error
  }

  return sameTarget(next, target) ? target : next
}

const sameTarget = (next: ConnectionTarget, target: ConnectionTarget): boolean =>
  next.catalog === target.catalog &&
  next.connectUrl === target.connectUrl &&
  next.connectionId === target.connectionId &&
  next.alias === target.alias &&
  next.detail === target.detail &&
  next.instructions === target.instructions &&
  next.discoveryError === target.discoveryError &&
  next.state === target.state &&
  next.tools.length === target.tools.length &&
  next.tools.every((tool, index) => tool === target.tools[index]) &&
  sameEnvFields(next.requiredEnv, target.requiredEnv)

// Every frame carries a fresh array, so identity would churn the row and remount its open inputs.
const sameEnvFields = (next: SetupField[], previous: SetupField[]): boolean =>
  next.length === previous.length &&
  next.every(
    (field, index) =>
      field.name === previous[index].name &&
      field.prompt === previous[index].prompt &&
      field.required === previous[index].required &&
      field.secret === previous[index].secret &&
      field.default === previous[index].default
  )

/** Apply one `connection.update` frame. Every frame carries the operation's full target snapshot, so
 *  the store overlays it; frames for another operation or for a settled request are ignored. */
export function applyConnectionUpdate(request: ConnectionRequest, update: ConnectionUpdatePayload): ConnectionRequest {
  if (update.op_id !== request.opId || request.settled) {
    return request
  }

  return applyOperationStatus(request, update)
}

export function setConnectionRequest(request: ConnectionRequest): void {
  const key = keyFor(request.sessionId)
  const requests = $connectionRequests.get()
  const previous = requests[key]

  if (previous && previous.opId !== request.opId) {
    const catalog = isCatalogRequest(request)
    const stays = (entry: ConnectionRequest) => isCatalogRequest(entry) || isCatalogRequest(entry) !== catalog
    const kept = ($keptConnectionRequests.get()[key] ?? []).filter(entry => entry.opId !== request.opId && stays(entry))

    $keptConnectionRequests.set({
      ...$keptConnectionRequests.get(),
      [key]: previous.settled && stays(previous) ? [...kept, previous] : kept
    })
  }

  $connectionRequests.set({ ...requests, [key]: request })
}

export function updateConnectionRequest(sessionId: string | null, update: ConnectionUpdatePayload): void {
  const current = $connectionRequests.get()[keyFor(sessionId)]

  if (!current) {
    return
  }

  const next = applyConnectionUpdate(current, update)

  if (next !== current) {
    setConnectionRequest(next)
  }
}

/** Drop the session's operation (or, with no session, every one) whose op id matches, from the current
 *  and the replaced caches alike. */
export function clearConnectionRequest(opId?: string, sessionId?: string | null): void {
  const cleared = ([key, value]: [string, ConnectionRequest]) =>
    (sessionId === undefined || key === keyFor(sessionId)) && (!opId || value.opId === opId)

  const entries = Object.entries($connectionRequests.get())
  const current = entries.filter(entry => !cleared(entry))

  if (current.length !== entries.length) {
    $connectionRequests.set(Object.fromEntries(current))
  }

  const keptBefore = $keptConnectionRequests.get()

  const keptAfter = Object.fromEntries(
    Object.entries(keptBefore).map(([key, list]) => [key, list.filter(request => !cleared([key, request]))])
  )

  if (Object.keys(keptBefore).some(key => keptAfter[key].length !== keptBefore[key].length)) {
    $keptConnectionRequests.set(keptAfter)
  }
}

/** The composer's Enter handler reads this without subscribing. */
export const hasConnectionRequest = (sessionId: string | null | undefined): boolean => {
  const request = $connectionRequests.get()[keyFor(sessionId)]

  return Boolean(request && !request.settled)
}

export async function connectionOwnerFor(sessionId: string, method: string): Promise<ConnectionOwner | null> {
  const ambientProfile = $activeGatewayProfile.get()

  try {
    const scope = await resolveSessionOwner(sessionId)
    assertSessionOwnerResolved(scope, { method, sessionId })

    return {
      connectionId: isSessionOwnerRoute(scope) ? scope.connectionId : null,
      profile: isSessionOwnerRoute(scope) ? scope.profile : scope || ambientProfile
    }
  } catch {
    return null
  }
}

export const connectionRequestOpen = (
  request: ConnectionRequest
): request is ConnectionRequest & { sessionId: string } => {
  const current = $connectionRequests.get()[keyFor(request.sessionId)]

  return Boolean(request.sessionId && current && current.opId === request.opId && !current.settled)
}

/** Drive the operation. The entry stays in the store: the backend answers with `connection.update`
 *  and the card re-renders from that; only settlement removes it. */
export async function respondToConnectionRequest(
  request: ConnectionRequest,
  outcome: ConnectionAnswer
): Promise<boolean> {
  if (!connectionRequestOpen(request)) {
    return false
  }

  const params = {
    op_id: request.opId,
    owner: { session_id: request.sessionId, type: 'session' as const },
    result: outcome
  }

  const owner = await connectionOwnerFor(request.sessionId, 'connection.respond')

  if (owner) {
    await requestGatewayForAgent(owner.connectionId, owner.profile, 'connection.respond', params)
  } else {
    await $gateway.get()?.request('connection.respond', params)
  }

  return true
}

/** Not now on one target. */
export const skipConnectionTarget = (request: ConnectionRequest, name: string): Promise<boolean> =>
  respondToConnectionRequest(request, { targets: [{ name, status: 'skipped' }] })

/** Continue: end the operation now with whatever is unresolved. */
export const continueConnectionRequest = (request: ConnectionRequest): Promise<boolean> =>
  respondToConnectionRequest(request, { settled_by: 'continue' })

// Typing a message while the card is open ends the operation, otherwise the typed message waits behind
// the blocked tool until the deadline.
export async function skipConnectionRequest(sessionId: string | null | undefined): Promise<boolean> {
  const request = $connectionRequests.get()[keyFor(sessionId)]

  if (!request || request.settled) {
    return false
  }

  try {
    await continueConnectionRequest(request)
  } catch {
    // A failed skip must not block the message; the tool settles at its deadline.
  }

  return true
}
