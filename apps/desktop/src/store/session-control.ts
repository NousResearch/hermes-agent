import { atom } from 'nanostores'

import { $gateway } from './gateway'
import { refreshSessionGoal } from './goals'
import { isSessionGone, isSessionGoneForBackgroundPolling, markSessionGone } from './runtime-gone'
import { ambientRequestFor } from './session-gone-latch'
import { requestForOwnedSession } from './session-states'

export type SessionControlGoalStatus = 'active' | 'done' | 'paused'
export type SessionControlLoopMode = 'interval' | 'self_paced'
export type SessionControlLoopStatus = 'active' | 'done' | 'paused'
export type SessionControlHeartbeatStatus = 'active' | 'paused'

export interface SessionControlGoalContract {
  boundaries: string
  constraints: string
  outcome: string
  stop_when: string
  verification: string
}

export interface SessionControlGate {
  attempts: number
  command: string
  last_exit_code: number | null
  max_retries: number
  timeout_seconds: number
}

export type SessionControlWaitBarrier =
  | { reason: string; type: 'until'; until_at: number }
  | { reason: string; target: string; type: 'session' }
  | { reason: string; target: number; type: 'pid' }

export interface SessionControlGoal {
  contract: SessionControlGoalContract
  gates: SessionControlGate[]
  last_reason?: string
  last_verdict?: 'blocked' | 'continue' | 'done' | 'skipped' | 'wait'
  max_turns: number
  paused_reason?: string
  status: SessionControlGoalStatus
  subgoals: string[]
  title: string
  turns_used: number
  updated_at?: number
  wait_barrier?: SessionControlWaitBarrier
  created_at?: number
}

export interface SessionControlLoop {
  awaiting_response: boolean
  created_at: number
  current_delay: number
  deferred_by_goal: boolean
  interval_seconds: number
  last_fired_at: number
  last_stop_reason?: string
  max_ticks: number
  mode: SessionControlLoopMode
  next_due_at: number
  paused_reason?: string
  prompt: string
  status: SessionControlLoopStatus
  ticks_fired: number
  times: number
  until: string
}

export interface SessionControlHeartbeat {
  created_at: number
  fire_count: number
  interval_seconds: number
  last_fired_at: number
  prompt: string
  status: SessionControlHeartbeatStatus
}

export interface SessionControlSnapshot {
  goal: SessionControlGoal | null
  heartbeat: SessionControlHeartbeat | null
  loop: SessionControlLoop | null
  revision: string
  updated_at: number
}

export type SessionControlAction =
  | 'goal.clear'
  | 'goal.pause'
  | 'goal.resume'
  | 'goal.unwait'
  | 'heartbeat.clear'
  | 'heartbeat.pause'
  | 'heartbeat.resume'
  | 'loop.pause'
  | 'loop.resume'
  | 'loop.stop'
  | 'subgoal.add'
  | 'subgoal.clear'
  | 'subgoal.remove'

export type SessionControlActionArgs = { index: number } | { text: string }

export interface SessionControlDispatch {
  display: string | null
  message: string | null
  notice: string | null
  output: string | null
  type: 'exec' | 'send'
}

export interface SessionControlEntry {
  capability: 'unknown' | 'supported' | 'unsupported'
  error: string | null
  loading: boolean
  pendingAction: SessionControlAction | null
  snapshot: SessionControlSnapshot | null
}

interface RefreshOptions {
  background?: boolean
}

type UnknownRecord = Record<string, unknown>

const GOAL_STATUSES = new Set<SessionControlGoalStatus>(['active', 'done', 'paused'])

const GOAL_VERDICTS = new Set<NonNullable<SessionControlGoal['last_verdict']>>([
  'blocked',
  'continue',
  'done',
  'skipped',
  'wait'
])

const LOOP_MODES = new Set<SessionControlLoopMode>(['interval', 'self_paced'])
const LOOP_STATUSES = new Set<SessionControlLoopStatus>(['active', 'done', 'paused'])
const HEARTBEAT_STATUSES = new Set<SessionControlHeartbeatStatus>(['active', 'paused'])
const DISPATCH_TYPES = new Set<SessionControlDispatch['type']>(['exec', 'send'])
const ERROR_LIMIT = 240

const versions = new Map<string, number>()
const eventVersions = new Map<string, number>()

export const $sessionControlBySession = atom<Record<string, SessionControlEntry>>({})

function isRecord(value: unknown): value is UnknownRecord {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
}

function hasOwn(value: UnknownRecord, key: string): boolean {
  return Object.prototype.hasOwnProperty.call(value, key)
}

function hasExactFields(value: UnknownRecord, required: string[], optional: string[] = []): boolean {
  const allowed = new Set([...required, ...optional])

  return required.every(key => hasOwn(value, key)) && Object.keys(value).every(key => allowed.has(key))
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value)
}

function isInteger(value: unknown): value is number {
  return Number.isInteger(value)
}

function hasOptionalStrings(value: UnknownRecord, keys: string[]): boolean {
  return keys.every(key => !hasOwn(value, key) || typeof value[key] === 'string')
}

function parseGoalContract(value: unknown): SessionControlGoalContract | null {
  if (
    !isRecord(value) ||
    !hasExactFields(value, ['outcome', 'verification', 'constraints', 'boundaries', 'stop_when'])
  ) {
    return null
  }

  if (
    typeof value.outcome !== 'string' ||
    typeof value.verification !== 'string' ||
    typeof value.constraints !== 'string' ||
    typeof value.boundaries !== 'string' ||
    typeof value.stop_when !== 'string'
  ) {
    return null
  }

  return {
    boundaries: value.boundaries,
    constraints: value.constraints,
    outcome: value.outcome,
    stop_when: value.stop_when,
    verification: value.verification
  }
}

function parseGate(value: unknown): SessionControlGate | null {
  if (
    !isRecord(value) ||
    !hasExactFields(value, ['command', 'timeout_seconds', 'max_retries', 'attempts', 'last_exit_code'])
  ) {
    return null
  }

  if (
    typeof value.command !== 'string' ||
    !isInteger(value.timeout_seconds) ||
    !isInteger(value.max_retries) ||
    !isInteger(value.attempts) ||
    (value.last_exit_code !== null && !isInteger(value.last_exit_code))
  ) {
    return null
  }

  return {
    attempts: value.attempts,
    command: value.command,
    last_exit_code: value.last_exit_code,
    max_retries: value.max_retries,
    timeout_seconds: value.timeout_seconds
  }
}

function parseWaitBarrier(value: unknown): SessionControlWaitBarrier | null {
  if (!isRecord(value) || typeof value.type !== 'string') {
    return null
  }

  if (value.type === 'until') {
    if (
      !hasExactFields(value, ['type', 'until_at', 'reason']) ||
      !isFiniteNumber(value.until_at) ||
      typeof value.reason !== 'string'
    ) {
      return null
    }

    return { reason: value.reason, type: 'until', until_at: value.until_at }
  }

  if (value.type === 'session') {
    if (
      !hasExactFields(value, ['type', 'target', 'reason']) ||
      typeof value.target !== 'string' ||
      typeof value.reason !== 'string'
    ) {
      return null
    }

    return { reason: value.reason, target: value.target, type: 'session' }
  }

  if (value.type === 'pid') {
    if (
      !hasExactFields(value, ['type', 'target', 'reason']) ||
      !isInteger(value.target) ||
      typeof value.reason !== 'string'
    ) {
      return null
    }

    return { reason: value.reason, target: value.target, type: 'pid' }
  }

  return null
}

function parseGoal(value: unknown): SessionControlGoal | null {
  const required = ['title', 'status', 'turns_used', 'max_turns', 'contract', 'subgoals', 'gates']
  const optional = ['created_at', 'updated_at', 'paused_reason', 'last_verdict', 'last_reason', 'wait_barrier']

  if (!isRecord(value) || !hasExactFields(value, required, optional)) {
    return null
  }

  if (
    typeof value.title !== 'string' ||
    typeof value.status !== 'string' ||
    !GOAL_STATUSES.has(value.status as SessionControlGoalStatus) ||
    !isInteger(value.turns_used) ||
    !isInteger(value.max_turns) ||
    !Array.isArray(value.subgoals) ||
    !value.subgoals.every(subgoal => typeof subgoal === 'string') ||
    !Array.isArray(value.gates) ||
    !hasOptionalStrings(value, ['paused_reason', 'last_reason']) ||
    (hasOwn(value, 'created_at') && !isFiniteNumber(value.created_at)) ||
    (hasOwn(value, 'updated_at') && !isFiniteNumber(value.updated_at)) ||
    (hasOwn(value, 'last_verdict') &&
      (typeof value.last_verdict !== 'string' ||
        !GOAL_VERDICTS.has(value.last_verdict as NonNullable<SessionControlGoal['last_verdict']>)))
  ) {
    return null
  }

  const contract = parseGoalContract(value.contract)
  const parsedGates = value.gates.map(parseGate)
  const waitBarrier = hasOwn(value, 'wait_barrier') ? parseWaitBarrier(value.wait_barrier) : undefined

  if (!contract || parsedGates.some(gate => gate === null) || (hasOwn(value, 'wait_barrier') && !waitBarrier)) {
    return null
  }

  const gates = parsedGates.filter((gate): gate is SessionControlGate => gate !== null)

  const goal: SessionControlGoal = {
    contract,
    gates,
    max_turns: value.max_turns,
    status: value.status as SessionControlGoalStatus,
    subgoals: [...value.subgoals],
    title: value.title,
    turns_used: value.turns_used
  }

  if (hasOwn(value, 'created_at')) {
    goal.created_at = value.created_at as number
  }

  if (hasOwn(value, 'updated_at')) {
    goal.updated_at = value.updated_at as number
  }

  if (hasOwn(value, 'paused_reason')) {
    goal.paused_reason = value.paused_reason as string
  }

  if (hasOwn(value, 'last_reason')) {
    goal.last_reason = value.last_reason as string
  }

  if (hasOwn(value, 'last_verdict')) {
    goal.last_verdict = value.last_verdict as SessionControlGoal['last_verdict']
  }

  if (waitBarrier) {
    goal.wait_barrier = waitBarrier
  }

  return goal
}

function parseLoop(value: unknown): SessionControlLoop | null {
  const required = [
    'prompt',
    'status',
    'mode',
    'interval_seconds',
    'current_delay',
    'times',
    'until',
    'max_ticks',
    'ticks_fired',
    'created_at',
    'last_fired_at',
    'next_due_at',
    'awaiting_response',
    'deferred_by_goal'
  ]

  if (!isRecord(value) || !hasExactFields(value, required, ['paused_reason', 'last_stop_reason'])) {
    return null
  }

  if (
    typeof value.prompt !== 'string' ||
    typeof value.status !== 'string' ||
    !LOOP_STATUSES.has(value.status as SessionControlLoopStatus) ||
    typeof value.mode !== 'string' ||
    !LOOP_MODES.has(value.mode as SessionControlLoopMode) ||
    !isFiniteNumber(value.interval_seconds) ||
    !isFiniteNumber(value.current_delay) ||
    !isInteger(value.times) ||
    typeof value.until !== 'string' ||
    !isInteger(value.max_ticks) ||
    !isInteger(value.ticks_fired) ||
    !isFiniteNumber(value.created_at) ||
    !isFiniteNumber(value.last_fired_at) ||
    !isFiniteNumber(value.next_due_at) ||
    typeof value.awaiting_response !== 'boolean' ||
    typeof value.deferred_by_goal !== 'boolean' ||
    !hasOptionalStrings(value, ['paused_reason', 'last_stop_reason'])
  ) {
    return null
  }

  const loop: SessionControlLoop = {
    awaiting_response: value.awaiting_response,
    created_at: value.created_at,
    current_delay: value.current_delay,
    deferred_by_goal: value.deferred_by_goal,
    interval_seconds: value.interval_seconds,
    last_fired_at: value.last_fired_at,
    max_ticks: value.max_ticks,
    mode: value.mode as SessionControlLoopMode,
    next_due_at: value.next_due_at,
    prompt: value.prompt,
    status: value.status as SessionControlLoopStatus,
    ticks_fired: value.ticks_fired,
    times: value.times,
    until: value.until
  }

  if (hasOwn(value, 'paused_reason')) {
    loop.paused_reason = value.paused_reason as string
  }

  if (hasOwn(value, 'last_stop_reason')) {
    loop.last_stop_reason = value.last_stop_reason as string
  }

  return loop
}

function parseHeartbeat(value: unknown): SessionControlHeartbeat | null {
  const required = ['prompt', 'status', 'interval_seconds', 'created_at', 'last_fired_at', 'fire_count']

  if (!isRecord(value) || !hasExactFields(value, required)) {
    return null
  }

  if (
    typeof value.prompt !== 'string' ||
    typeof value.status !== 'string' ||
    !HEARTBEAT_STATUSES.has(value.status as SessionControlHeartbeatStatus) ||
    !isInteger(value.interval_seconds) ||
    !isFiniteNumber(value.created_at) ||
    !isFiniteNumber(value.last_fired_at) ||
    !isInteger(value.fire_count)
  ) {
    return null
  }

  return {
    created_at: value.created_at,
    fire_count: value.fire_count,
    interval_seconds: value.interval_seconds,
    last_fired_at: value.last_fired_at,
    prompt: value.prompt,
    status: value.status as SessionControlHeartbeatStatus
  }
}

/** Parses the stable allowlisted backend shape into fresh renderer-owned data. */
export function parseSessionControlSnapshot(value: unknown): SessionControlSnapshot | null {
  if (!isRecord(value) || !hasExactFields(value, ['goal', 'loop', 'heartbeat', 'revision', 'updated_at'])) {
    return null
  }

  if (typeof value.revision !== 'string' || !isFiniteNumber(value.updated_at)) {
    return null
  }

  const goal = value.goal === null ? null : parseGoal(value.goal)
  const loop = value.loop === null ? null : parseLoop(value.loop)
  const heartbeat = value.heartbeat === null ? null : parseHeartbeat(value.heartbeat)

  if ((value.goal !== null && !goal) || (value.loop !== null && !loop) || (value.heartbeat !== null && !heartbeat)) {
    return null
  }

  return { goal, heartbeat, loop, revision: value.revision, updated_at: value.updated_at }
}

function parseSessionControlDispatch(value: unknown): SessionControlDispatch | null {
  if (!isRecord(value) || !hasExactFields(value, ['type', 'output', 'notice', 'message', 'display'])) {
    return null
  }

  if (
    typeof value.type !== 'string' ||
    !DISPATCH_TYPES.has(value.type as SessionControlDispatch['type']) ||
    ![value.output, value.notice, value.message, value.display].every(
      field => field === null || typeof field === 'string'
    )
  ) {
    return null
  }

  return {
    display: value.display as string | null,
    message: value.message as string | null,
    notice: value.notice as string | null,
    output: value.output as string | null,
    type: value.type as SessionControlDispatch['type']
  }
}

function emptyEntry(): SessionControlEntry {
  return { capability: 'unknown', error: null, loading: false, pendingAction: null, snapshot: null }
}

function currentVersion(sessionId: string): number {
  return versions.get(sessionId) ?? 0
}

function advanceVersion(sessionId: string): number {
  const next = currentVersion(sessionId) + 1
  versions.set(sessionId, next)

  return next
}

function currentEventVersion(sessionId: string): number {
  return eventVersions.get(sessionId) ?? 0
}

function advanceEventVersion(sessionId: string): number {
  const next = currentEventVersion(sessionId) + 1
  eventVersions.set(sessionId, next)

  return next
}

function isCurrent(sessionId: string, token: number): boolean {
  return currentVersion(sessionId) === token
}

function sameEntry(first: SessionControlEntry, second: SessionControlEntry): boolean {
  return (
    first.capability === second.capability &&
    first.error === second.error &&
    first.loading === second.loading &&
    first.pendingAction === second.pendingAction &&
    first.snapshot === second.snapshot
  )
}

function publishEntry(sessionId: string, next: SessionControlEntry): SessionControlEntry {
  const entries = $sessionControlBySession.get()
  const current = entries[sessionId]

  if (current && sameEntry(current, next)) {
    return current
  }

  $sessionControlBySession.set({ ...entries, [sessionId]: next })

  return next
}

function applyParsedSnapshot(sessionId: string, snapshot: SessionControlSnapshot): SessionControlEntry {
  advanceVersion(sessionId)
  const current = $sessionControlBySession.get()[sessionId] ?? emptyEntry()
  const nextSnapshot = current.snapshot?.revision === snapshot.revision ? current.snapshot : snapshot

  return publishEntry(sessionId, {
    capability: 'supported',
    error: null,
    loading: false,
    pendingAction: null,
    snapshot: nextSnapshot
  })
}

/** Applies a valid read/action snapshot and marks its session as supported. */
export function applySessionControlSnapshot(sessionId: string, rawSnapshot: unknown): SessionControlEntry | undefined {
  if (!sessionId) {
    return undefined
  }

  const snapshot = parseSessionControlSnapshot(rawSnapshot)

  return snapshot ? applyParsedSnapshot(sessionId, snapshot) : undefined
}

/** Applies an event update; invalid updates are deliberately a claimed no-op. */
export function applySessionControlUpdate(sessionId: string, rawSnapshot: unknown): SessionControlEntry | undefined {
  if (!sessionId) {
    return undefined
  }

  const snapshot = parseSessionControlSnapshot(rawSnapshot)

  if (!snapshot) {
    return undefined
  }

  const current = $sessionControlBySession.get()[sessionId] ?? emptyEntry()
  const actionIsPending = current.pendingAction !== null

  advanceEventVersion(sessionId)

  if (!actionIsPending) {
    advanceVersion(sessionId)
  }

  const nextSnapshot = current.snapshot?.revision === snapshot.revision ? current.snapshot : snapshot

  return publishEntry(sessionId, {
    ...current,
    capability: 'supported',
    error: null,
    loading: actionIsPending ? current.loading : false,
    pendingAction: actionIsPending ? current.pendingAction : null,
    snapshot: nextSnapshot
  })
}

/** Drops one runtime session's entry when the session record is closed/deleted. */
export function clearSessionControl(sessionId: string): void {
  if (!sessionId) {
    return
  }

  advanceVersion(sessionId)
  const entries = $sessionControlBySession.get()

  if (!(sessionId in entries)) {
    return
  }

  const { [sessionId]: _removed, ...remaining } = entries
  $sessionControlBySession.set(remaining)
}

/**
 * Wipes every entry — the gateway-switch seam. Entries are keyed by runtime
 * session id and a different backend mints new ids, so nothing here can be
 * reused; in-flight reads/actions from the old backend are invalidated by the
 * version bump so a late response cannot repopulate the map.
 */
export function clearAllSessionControl(): void {
  for (const sessionId of new Set([...versions.keys(), ...Object.keys($sessionControlBySession.get())])) {
    advanceVersion(sessionId)
  }

  $sessionControlBySession.set({})
  versions.clear()
  eventVersions.clear()
}

function beginRead(sessionId: string, background: boolean): number {
  const token = advanceVersion(sessionId)
  const current = $sessionControlBySession.get()[sessionId] ?? emptyEntry()

  publishEntry(sessionId, {
    ...current,
    error: null,
    loading: background ? current.loading : true
  })

  return token
}

function beginAction(sessionId: string, action: SessionControlAction): number {
  const token = advanceVersion(sessionId)
  const current = $sessionControlBySession.get()[sessionId] ?? emptyEntry()

  publishEntry(sessionId, {
    ...current,
    error: null,
    loading: true,
    pendingAction: action
  })

  return token
}

function boundedError(error: unknown): string {
  const message =
    error instanceof Error
      ? error.message
      : isRecord(error) && typeof error.message === 'string'
        ? error.message
        : 'Session control request failed'

  return message.trim().slice(0, ERROR_LIMIT) || 'Session control request failed'
}

function publishFailure(sessionId: string, token: number, error: unknown, clearPendingAction: boolean): void {
  if (!isCurrent(sessionId, token)) {
    return
  }

  const current = $sessionControlBySession.get()[sessionId] ?? emptyEntry()
  publishEntry(sessionId, {
    ...current,
    error: boundedError(error),
    loading: false,
    pendingAction: clearPendingAction ? null : current.pendingAction
  })
}

function finishGoneRequest(sessionId: string, token: number, clearPendingAction: boolean): void {
  if (!isCurrent(sessionId, token)) {
    return
  }

  markSessionGone(sessionId)
  const current = $sessionControlBySession.get()[sessionId] ?? emptyEntry()
  publishEntry(sessionId, {
    ...current,
    loading: false,
    pendingAction: clearPendingAction ? null : current.pendingAction
  })
}

function markUnsupported(sessionId: string, token: number): boolean {
  if (!isCurrent(sessionId, token)) {
    return false
  }

  const current = $sessionControlBySession.get()[sessionId] ?? emptyEntry()

  if (current.capability === 'unsupported') {
    return false
  }

  advanceVersion(sessionId)
  publishEntry(sessionId, {
    ...current,
    capability: 'unsupported',
    error: null,
    loading: false,
    pendingAction: null
  })

  return true
}

function isMethodNotFound(error: unknown): boolean {
  if (isRecord(error) && error.code === -32601) {
    return true
  }

  const message =
    error instanceof Error ? error.message : isRecord(error) && typeof error.message === 'string' ? error.message : ''

  return message.toLowerCase().includes('method not found') || message.toLowerCase().includes('method-not-found')
}

/** Hydrates one session's structured controls; background refreshes never flash a loading state. */
export async function refreshSessionControl(
  sessionId: string,
  options: RefreshOptions = {}
): Promise<SessionControlEntry | undefined> {
  const existing = $sessionControlBySession.get()[sessionId]

  if (!sessionId || existing?.capability === 'unsupported' || isSessionGone(sessionId)) {
    return existing
  }

  const gateway = $gateway.get()

  if (!gateway) {
    return existing
  }

  const token = beginRead(sessionId, Boolean(options.background))

  try {
    const response = await requestForOwnedSession<unknown>(
      sessionId,
      ambientRequestFor(gateway),
      'session.control.read',
      { session_id: sessionId }
    )

    if (!isCurrent(sessionId, token)) {
      return $sessionControlBySession.get()[sessionId]
    }

    const snapshot = isRecord(response) ? parseSessionControlSnapshot(response.control) : null

    if (!snapshot) {
      publishFailure(sessionId, token, new Error('Invalid session.control.read response'), false)

      return $sessionControlBySession.get()[sessionId]
    }

    return applyParsedSnapshot(sessionId, snapshot)
  } catch (error) {
    if (!isCurrent(sessionId, token)) {
      return $sessionControlBySession.get()[sessionId]
    }

    if (isMethodNotFound(error)) {
      const transitioned = markUnsupported(sessionId, token)

      if (transitioned) {
        await refreshSessionGoal(sessionId)
      }

      return $sessionControlBySession.get()[sessionId]
    }

    if (isSessionGoneForBackgroundPolling(error)) {
      finishGoneRequest(sessionId, token, false)

      return $sessionControlBySession.get()[sessionId]
    }

    publishFailure(sessionId, token, error, false)

    return $sessionControlBySession.get()[sessionId]
  }
}

/** Runs an allowlisted backend control action; callers own any composer/UI dispatch. */
export async function runSessionControlAction(
  sessionId: string,
  action: SessionControlAction,
  args?: SessionControlActionArgs
): Promise<SessionControlDispatch> {
  if (!sessionId) {
    throw new Error('A session id is required for session control')
  }

  if (isSessionGone(sessionId)) {
    throw new Error('Session not found')
  }

  const gateway = $gateway.get()

  if (!gateway) {
    throw new Error('Session control gateway is unavailable')
  }

  const eventVersion = currentEventVersion(sessionId)
  const token = beginAction(sessionId, action)

  try {
    const response = await requestForOwnedSession<unknown>(sessionId, ambientRequestFor(gateway), 'session.control', {
      action,
      args: args ?? {},
      session_id: sessionId
    })

    const snapshot = isRecord(response) ? parseSessionControlSnapshot(response.control) : null
    const dispatch = isRecord(response) ? parseSessionControlDispatch(response.dispatch) : null

    if (!snapshot || !dispatch) {
      const error = new Error('Invalid session.control action response')
      publishFailure(sessionId, token, error, true)
      throw error
    }

    if (isCurrent(sessionId, token)) {
      applyParsedSnapshot(sessionId, snapshot)
    }

    return dispatch
  } catch (error) {
    if (isMethodNotFound(error)) {
      const transitioned = eventVersion === currentEventVersion(sessionId) ? markUnsupported(sessionId, token) : false

      if (transitioned) {
        await refreshSessionGoal(sessionId)
      } else {
        publishFailure(sessionId, token, error, true)
      }
    } else if (isSessionGoneForBackgroundPolling(error)) {
      finishGoneRequest(sessionId, token, true)
    } else {
      publishFailure(sessionId, token, error, true)
    }

    throw error
  }
}

/** Refreshes only sessions already proven to support the structured-control RPC. */
export async function refreshSupportedSessionControlAfterTurn(sessionId: string): Promise<void> {
  if (!sessionId || $sessionControlBySession.get()[sessionId]?.capability !== 'supported') {
    return
  }

  await refreshSessionControl(sessionId, { background: true })
}
