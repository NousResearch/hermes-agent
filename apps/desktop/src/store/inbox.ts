import { atom } from 'nanostores'

import { getApiRequestConnection } from '@/hermes'

import { $gateway } from './gateway'

/**
 * Action Center (formerly Agent Inbox) — aggregation + navigation over the ACTIVE
 * connection + profile's persisted automation and live pending requests.
 *
 * The backend (`tui_gateway/methods_inbox.py`) is the single authoritative
 * source: it returns the same allowed goal/loop/heartbeat snapshots
 * `session.control.read` uses (so sessions that are NOT open are still
 * covered, without resuming or hydrating transcripts) plus redacted pending
 * approval / live clarify state. This store only reflects what the backend
 * published — it never resolves anything on open/dismiss, never fabricates a
 * snapshot, and rejects late responses after a gateway/profile switch (A-B-A).
 */

export type InboxLane = 'needs_you' | 'running' | 'waiting' | 'scheduled'
export type InboxBadge = 'amber' | 'none' | 'red'
export type InboxCapability = 'unknown' | 'supported' | 'unsupported'

const LANES = new Set<InboxLane>(['needs_you', 'running', 'waiting', 'scheduled'])
const BADGES = new Set<InboxBadge>(['amber', 'none', 'red'])

export interface InboxCoverage {
  approval_scope: string
  clarify_scope: string
  connection_scope: string
  errors: string[]
  partial: boolean
  profile: string
  scanned_sessions: number
}

export interface InboxPendingApproval {
  command_redacted: boolean
  count: number
  description: string
}

export interface InboxPendingClarify {
  count: number
}

export interface InboxItem {
  background_task_count: number
  background_task_count_unavailable: boolean
  categories: string[]
  cwd: string
  heartbeat: Record<string, unknown> | null
  goal: Record<string, unknown> | null
  lanes: InboxLane[]
  loop: Record<string, unknown> | null
  pending_approval: InboxPendingApproval | null
  pending_clarify: InboxPendingClarify | null
  /** Requests that ended without an answer and are still awaiting a redo/dismiss decision. */
  expired_request_count: number
  session_key: string
  source: string
  subagent_count: number
  subagent_count_unavailable: boolean
  title: string
}

export interface InboxCounts {
  needs_you: number
  running: number
  scheduled: number
  total: number
  waiting: number
}

export interface InboxSnapshot {
  badge: InboxBadge
  counts: InboxCounts
  coverage: InboxCoverage
  items: InboxItem[]
}

export interface InboxEntry {
  capability: InboxCapability
  error: string | null
  loading: boolean
  snapshot: InboxSnapshot | null
}

// ── request detail types (inbox.requests) ─────────────────────────────────────

export type InboxCategory = 'all' | 'goals' | 'loops' | 'heartbeats' | 'background_tasks' | 'subagents' | 'other'

export interface InboxRequestApproval {
  allow_permanent: boolean | null
  allow_session: boolean | null
  choices: string[]
  command: string
  description: string
  request_id: string
  smart_denied: boolean | null
  tool_name: string | null
}

export interface InboxRequestClarifyQuestion {
  multi_select: boolean
  qid: string
  question: string
  choices: string[] | null
}

export interface InboxRequestClarifyParams {
  answers: Record<string, string> | null
  choices: string[] | null
  multi_select: boolean | null
  question: string | null
  questions: InboxRequestClarifyQuestion[] | null
}

export interface InboxRequestClarification {
  kind: string
  params: InboxRequestClarifyParams
  request_id: string
}

export interface InboxRequestContextMessage {
  role: string
  text: string
  timestamp: number | null
}

/**
 * Bounded, redacted excerpt of the owning session's recent turns, so a request can be
 * judged without leaving the panel. `available: false` carries a reason: an absent
 * excerpt must never read as "nothing was happening".
 */
export interface InboxRequestContext {
  available: boolean
  reason: string | null
  messages: InboxRequestContextMessage[]
}

/**
 * One request that ended without an answer. Kept (durably, gateway-side) so the panel can
 * still say what it was for and offer a redo — an expired request must never vanish silently.
 */
export interface InboxExpiredRequest {
  command: string
  description: string
  ended_at: number
  kind: string
  outcome: string
  request_id: string
}

export interface InboxRequestSessionDetail {
  approvals: InboxRequestApproval[]
  clarifications: InboxRequestClarification[]
  // Optional at the renderer boundary: an older gateway that predates the excerpt must
  // keep parsing, and the panel reports the absence instead of an empty transcript.
  context?: InboxRequestContext
  // Optional at the renderer boundary for the same reason: an older gateway has no
  // expired-request store at all.
  expired_requests?: InboxExpiredRequest[]
  live_session_ids: string[]
}

export interface InboxRequestsCoverage {
  approval_count: number
  clarification_count: number
  context_anchor: string
  errors: string[]
  live_session_count: number
  profile: string
  session_key: string
}

export interface InboxRequestDetails {
  coverage: InboxRequestsCoverage
  sessions: InboxRequestSessionDetail[]
}

export interface InboxRequestDetailsEntry {
  details: InboxRequestDetails | null
  error: string | null
  loading: boolean
}

type UnknownRecord = Record<string, unknown>

const isRecord = (value: unknown): value is UnknownRecord =>
  value !== null && typeof value === 'object' && !Array.isArray(value)

const isStringArray = (value: unknown): value is string[] =>
  Array.isArray(value) && value.every(item => typeof item === 'string')

const asString = (value: unknown): string => (typeof value === 'string' ? value : '')

function parseCoverage(value: unknown): InboxCoverage | null {
  if (!isRecord(value)) {return null}
  const errors = value.errors

  if (!Array.isArray(errors) || !errors.every(item => typeof item === 'string')) {return null}

  if (
    typeof value.approval_scope !== 'string' ||
    typeof value.clarify_scope !== 'string' ||
    typeof value.connection_scope !== 'string' ||
    typeof value.profile !== 'string' ||
    typeof value.partial !== 'boolean' ||
    typeof value.scanned_sessions !== 'number'
  ) {
    return null
  }

  return {
    approval_scope: value.approval_scope,
    clarify_scope: value.clarify_scope,
    connection_scope: value.connection_scope,
    errors: errors as string[],
    partial: value.partial,
    profile: value.profile,
    scanned_sessions: value.scanned_sessions
  }
}

function parseCounts(value: unknown): InboxCounts | null {
  if (!isRecord(value)) {return null}

  for (const key of ['needs_you', 'running', 'scheduled', 'total', 'waiting'] as const) {
    if (typeof value[key] !== 'number') {return null}
  }

  return {
    needs_you: value.needs_you as number,
    running: value.running as number,
    scheduled: value.scheduled as number,
    total: value.total as number,
    waiting: value.waiting as number
  }
}

function parsePendingApproval(value: unknown): InboxPendingApproval | null {
  if (!isRecord(value) || typeof value.count !== 'number' || typeof value.description !== 'string') {return null}

  return { command_redacted: value.command_redacted !== false, count: value.count, description: value.description }
}

function parsePendingClarify(value: unknown): InboxPendingClarify | null {
  if (!isRecord(value) || typeof value.count !== 'number') {return null}

  return { count: value.count }
}

function parseItem(value: unknown): InboxItem | null {
  if (!isRecord(value) || !isStringArray(value.lanes)) {return null}

  if (!value.lanes.every(lane => LANES.has(lane as InboxLane))) {return null}

  const pendingApproval =
    value.pending_approval === null || value.pending_approval === undefined
      ? null
      : parsePendingApproval(value.pending_approval)

  const pendingClarify =
    value.pending_clarify === null || value.pending_clarify === undefined
      ? null
      : parsePendingClarify(value.pending_clarify)

  if (
    (value.pending_approval !== null && value.pending_approval !== undefined && !pendingApproval) ||
    (value.pending_clarify !== null && value.pending_clarify !== undefined && !pendingClarify)
  ) {
    return null
  }

  const categories = Array.isArray(value.categories) && value.categories.every(c => typeof c === 'string')
    ? (value.categories as string[])
    : []

  return {
    background_task_count: typeof value.background_task_count === 'number' ? value.background_task_count : 0,
    background_task_count_unavailable: value.background_task_count_unavailable === true,
    categories,
    cwd: asString(value.cwd),
    expired_request_count: typeof value.expired_request_count === 'number' ? value.expired_request_count : 0,
    goal: value.goal === null ? null : isRecord(value.goal) ? value.goal : null,
    heartbeat: value.heartbeat === null ? null : isRecord(value.heartbeat) ? value.heartbeat : null,
    lanes: value.lanes as InboxLane[],
    loop: value.loop === null ? null : isRecord(value.loop) ? value.loop : null,
    pending_approval: pendingApproval,
    pending_clarify: pendingClarify,
    session_key: asString(value.session_key),
    source: asString(value.source),
    subagent_count: typeof value.subagent_count === 'number' ? value.subagent_count : 0,
    subagent_count_unavailable: value.subagent_count_unavailable === true,
    title: asString(value.title)
  }
}

function parseSnapshot(value: unknown): InboxSnapshot | null {
  if (!isRecord(value)) {return null}
  const coverage = parseCoverage(value.coverage)
  const counts = parseCounts(value.counts)

  if (!coverage || !counts || !BADGES.has(value.badge as InboxBadge)) {return null}

  if (!Array.isArray(value.items)) {return null}
  const items: InboxItem[] = []

  for (const item of value.items) {
    const parsed = parseItem(item)

    if (!parsed) {return null}
    items.push(parsed)
  }

  return { badge: value.badge as InboxBadge, counts, coverage, items }
}

/** Parses the allowlisted backend envelope into renderer-owned data. */
export function parseInboxSnapshot(value: unknown): InboxSnapshot | null {
  if (!isRecord(value)) {return null}

  return parseSnapshot(value.inbox)
}

// ── store + A-B-A request tokens (mirrors store/cron.ts) ─────────────────────
export const $inbox = atom<InboxEntry>({ capability: 'unknown', error: null, loading: false, snapshot: null })

/** An inbox request is scoped to the gateway connection + profile, so a switch
 *  invalidates in-flight requests from the old backend. */
function inboxRequestScope(profile: string): string {
  return `${getApiRequestConnection() ?? ''}\u0000${profile}`
}

export interface InboxScopeToken {
  generation: number
  scope: string
}

let inboxScope = ''
let inboxScopeGeneration = 0
let inboxRequestGeneration = 0
let inboxRequestInFlight: InboxScopeToken | null = null

function activateInboxScope(scope: string): void {
  if (scope === inboxScope) {return}
  inboxScope = scope
  inboxScopeGeneration += 1
  inboxRequestGeneration += 1
}

export function beginInboxRequest(scope: string): InboxScopeToken {
  activateInboxScope(scope)
  inboxRequestGeneration += 1

  return { generation: inboxRequestGeneration, scope }
}

export function isInboxRequestCurrent(token: InboxScopeToken): boolean {
  return token.scope === inboxScope && token.generation === inboxRequestGeneration
}

export function invalidateInboxRequests(): void {
  inboxRequestGeneration += 1
  inboxScopeGeneration += 1
}

function publishEntry(next: InboxEntry): void {
  const current = $inbox.get()

  if (
    current.capability === next.capability &&
    current.error === next.error &&
    current.loading === next.loading &&
    current.snapshot === next.snapshot
  ) {
    return
  }

  $inbox.set(next)
}

export function commitInboxRequest(token: InboxScopeToken, snapshot: InboxSnapshot): boolean {
  if (!isInboxRequestCurrent(token)) {return false}
  // Consume the token so neither a duplicate completion nor any older request
  // can publish after this authoritative snapshot.
  inboxRequestGeneration += 1
  publishEntry({ capability: 'supported', error: null, loading: false, snapshot })

  return true
}

/** Wipes the inbox on a gateway/profile switch: a different backend mints a
 *  fresh scope, and the generation bump rejects any in-flight late response. */
export function clearInbox(): void {
  invalidateInboxRequests()
  inboxRequestInFlight = null
  publishEntry({ capability: 'unknown', error: null, loading: false, snapshot: null })
  clearAllRequestDetails()
}

// ── request details (inbox.requests) ──────────────────────────────────────────

export const $inboxRequestDetails = atom<Record<string, InboxRequestDetailsEntry>>({})

let detailRequestGeneration = 0

function publishRequestDetails(key: string, entry: InboxRequestDetailsEntry): void {
  const all = $inboxRequestDetails.get()
  const prev = all[key]

  if (prev && prev.loading === entry.loading && prev.error === entry.error && prev.details === entry.details) {return}
  $inboxRequestDetails.set({ ...all, [key]: entry })
}

export function clearAllRequestDetails(): void {
  detailRequestGeneration += 1
  $inboxRequestDetails.set({})
}

function parseRequestApproval(value: unknown): InboxRequestApproval | null {
  if (!isRecord(value)) {return null}

  return {
    allow_permanent: typeof value.allow_permanent === 'boolean' ? value.allow_permanent : null,
    allow_session: typeof value.allow_session === 'boolean' ? value.allow_session : null,
    choices: Array.isArray(value.choices) ? (value.choices as string[]) : [],
    command: asString(value.command),
    description: asString(value.description),
    request_id: asString(value.request_id),
    smart_denied: typeof value.smart_denied === 'boolean' ? value.smart_denied : null,
    tool_name: typeof value.tool_name === 'string' ? value.tool_name : null
  }
}

function parseClarifyQuestion(value: unknown): InboxRequestClarifyQuestion | null {
  if (!isRecord(value)) {return null}

  return {
    multi_select: value.multi_select === true,
    qid: asString(value.qid),
    question: asString(value.question),
    choices: Array.isArray(value.choices) ? (value.choices as string[]) : null
  }
}

function parseClarifyParams(value: unknown): InboxRequestClarifyParams | null {
  if (!isRecord(value)) {return null}
  const rawQuestions = value.questions

  const questions = Array.isArray(rawQuestions)
    ? rawQuestions.map(parseClarifyQuestion).filter((q): q is InboxRequestClarifyQuestion => q !== null)
    : null

  return {
    answers: isRecord(value.answers) ? (value.answers as Record<string, string>) : null,
    choices: Array.isArray(value.choices) ? (value.choices as string[]) : null,
    multi_select: typeof value.multi_select === 'boolean' ? value.multi_select : null,
    question: typeof value.question === 'string' ? value.question : null,
    questions
  }
}

function parseClarification(value: unknown): InboxRequestClarification | null {
  if (!isRecord(value)) {return null}
  const params = parseClarifyParams(value.params)

  if (!params) {return null}

  return {
    kind: asString(value.kind) || 'single',
    params,
    request_id: asString(value.request_id)
  }
}

function parseContextMessage(value: unknown): InboxRequestContextMessage | null {
  if (!isRecord(value)) {return null}

  const text = asString(value.text)

  if (!text) {return null}

  return {
    role: asString(value.role),
    text,
    timestamp: typeof value.timestamp === 'number' ? value.timestamp : null
  }
}

function parseRequestContext(value: unknown): InboxRequestContext {
  // A backend that predates this field reports nothing. Say so out loud rather than
  // rendering an empty excerpt as if the session were quiet.
  if (!isRecord(value)) {
    return { available: false, reason: 'context not reported by this backend', messages: [] }
  }

  const messages = Array.isArray(value.messages)
    ? (value.messages as unknown[]).map(parseContextMessage).filter((m): m is InboxRequestContextMessage => m !== null)
    : []

  return {
    available: value.available === true,
    reason: typeof value.reason === 'string' && value.reason ? value.reason : null,
    messages
  }
}

function parseExpiredRequest(value: unknown): InboxExpiredRequest | null {
  if (!isRecord(value)) {return null}

  const requestId = asString(value.request_id)

  if (!requestId) {return null}

  return {
    command: asString(value.command),
    description: asString(value.description),
    ended_at: typeof value.ended_at === 'number' ? value.ended_at : 0,
    kind: asString(value.kind),
    outcome: asString(value.outcome),
    request_id: requestId
  }
}

function parseRequestSessionDetail(value: unknown): InboxRequestSessionDetail | null {
  if (!isRecord(value)) {return null}

  const approvals = Array.isArray(value.approvals)
    ? (value.approvals as unknown[]).map(parseRequestApproval).filter((a): a is InboxRequestApproval => a !== null)
    : []

  const clarifications = Array.isArray(value.clarifications)
    ? (value.clarifications as unknown[]).map(parseClarification).filter((c): c is InboxRequestClarification => c !== null)
    : []

  // Requests that died unanswered. Parsed like every other field — dropping them here means
  // the wire carries the record and the panel still shows nothing (live, 2026-09-18).
  const expired_requests = Array.isArray(value.expired_requests)
    ? (value.expired_requests as unknown[]).map(parseExpiredRequest).filter((e): e is InboxExpiredRequest => e !== null)
    : []

  const live_session_ids = Array.isArray(value.live_session_ids)
    ? (value.live_session_ids as unknown[]).filter((id): id is string => typeof id === 'string')
    : []

  return { approvals, clarifications, context: parseRequestContext(value.context), expired_requests, live_session_ids }
}

function parseRequestDetails(value: unknown): InboxRequestDetails | null {
  if (!isRecord(value)) {return null}

  const sessions = Array.isArray(value.sessions)
    ? (value.sessions as unknown[]).map(parseRequestSessionDetail).filter((s): s is InboxRequestSessionDetail => s !== null)
    : []

  const rawCoverage = value.coverage

  const coverage: InboxRequestsCoverage = isRecord(rawCoverage) ? {
    approval_count: typeof rawCoverage.approval_count === 'number' ? rawCoverage.approval_count : 0,
    clarification_count: typeof rawCoverage.clarification_count === 'number' ? rawCoverage.clarification_count : 0,
    context_anchor: asString(rawCoverage.context_anchor),
    errors: Array.isArray(rawCoverage.errors) ? (rawCoverage.errors as string[]) : [],
    live_session_count: typeof rawCoverage.live_session_count === 'number' ? rawCoverage.live_session_count : 0,
    profile: asString(rawCoverage.profile),
    session_key: asString(rawCoverage.session_key)
  } : {
    approval_count: 0, clarification_count: 0, context_anchor: '', errors: [],
    live_session_count: 0, profile: '', session_key: ''
  }

  return { coverage, sessions }
}

export async function fetchInboxRequestDetails(
  sessionKey: string,
  profile: string,
  request: InboxRequest = defaultInboxRequest
): Promise<InboxRequestDetails | null> {
  // Capture the generation at call time so a mid-flight clearInbox (scope
  // switch) causes the late response to be dropped instead of repopulating
  // with stale data from the old backend/profile.
  const generation = ++detailRequestGeneration
  publishRequestDetails(sessionKey, { details: null, error: null, loading: true })

  try {
    const response = await request('inbox.requests', { session_key: sessionKey, ...(profile ? { profile } : {}) })

    if (generation !== detailRequestGeneration) {
      // A scope switch happened while we were in flight — drop the result.
      return null
    }

    const details = parseRequestDetails(response)

    if (!details) {
      publishRequestDetails(sessionKey, { details: null, error: 'Invalid inbox.requests response', loading: false })

      return null
    }

    // Validate that the response belongs to the profile we asked for.
    if (profile && details.coverage.profile && details.coverage.profile !== profile) {
      publishRequestDetails(sessionKey, { details: null, error: 'Response from wrong profile', loading: false })

      return null
    }

    publishRequestDetails(sessionKey, { details, error: null, loading: false })

    return details
  } catch (error) {
    if (generation !== detailRequestGeneration) {
      return null
    }

    publishRequestDetails(sessionKey, { details: null, error: boundedError(error), loading: false })

    return null
  }
}

export interface ApprovalRespondResult {
  resolved: number
}

export async function respondToApproval(params: {
  choice: string
  liveSessionId: string
  profile?: string
  requestId?: string
  request?: InboxRequest
}): Promise<ApprovalRespondResult> {
  const { choice, liveSessionId, profile, requestId, request: rpc = defaultInboxRequest } = params

  const result = await rpc('approval.respond', {
    session_id: liveSessionId,
    choice,
    all: false,
    ...(requestId ? { request_id: requestId } : {}),
    ...(profile ? { profile } : {})
  }) as ApprovalRespondResult

  return result
}

export type InboxAutomationAction =
  | 'goal.pause'
  | 'goal.resume'
  | 'heartbeat.pause'
  | 'heartbeat.resume'
  | 'loop.pause'
  | 'loop.resume'

/**
 * Run one automation control action for the session behind the panel. Same `session.control`
 * allowlist the composer's status cards go through, so the panel can never exceed the chat's
 * authority. Refused (4009-class) when the session is not live — nothing is touched remotely.
 */
export async function runInboxAutomationAction(params: {
  action: InboxAutomationAction
  liveSessionId: string
  profile?: string
  request?: InboxRequest
}): Promise<void> {
  const { action, liveSessionId, profile, request: rpc = defaultInboxRequest } = params

  await rpc('session.control', {
    action,
    args: {},
    session_id: liveSessionId,
    ...(profile ? { profile } : {})
  })
}

export interface RedoExpiredResult {
  record_cleared: boolean
  redone: boolean
  session_id: string
}

/**
 * Re-raise an expired request: the gateway asks the session to attempt the action again,
 * which raises a fresh approval the panel can answer. Refused (4009) when the session is
 * not live — nothing is resumed on the operator's behalf.
 */
export async function redoExpiredRequest(params: {
  profile?: string
  requestId: string
  request?: InboxRequest
  sessionKey: string
}): Promise<RedoExpiredResult> {
  const { profile, requestId, request: rpc = defaultInboxRequest, sessionKey } = params

  return await rpc('inbox.redo', {
    session_key: sessionKey,
    request_id: requestId,
    ...(profile ? { profile } : {})
  }) as RedoExpiredResult
}

export interface DismissExpiredResult {
  dismissed: boolean
}

/** Drop one expired-request record — the operator's "I'm done with this one". */
export async function dismissExpiredRequest(params: {
  profile?: string
  requestId: string
  request?: InboxRequest
  sessionKey: string
}): Promise<DismissExpiredResult> {
  const { profile, requestId, request: rpc = defaultInboxRequest, sessionKey } = params

  return await rpc('inbox.dismiss', {
    session_key: sessionKey,
    request_id: requestId,
    ...(profile ? { profile } : {})
  }) as DismissExpiredResult
}

export interface ClarifyAnswerResult {
  status: string
}

export async function answerClarifySingle(params: {
  answer: string
  profile?: string
  requestId: string
  request?: InboxRequest
}): Promise<ClarifyAnswerResult> {
  const { answer, profile, requestId, request: rpc = defaultInboxRequest } = params

  const result = await rpc('request.answer', {
    id: requestId,
    result: { answer },
    ...(profile ? { profile } : {})
  }) as ClarifyAnswerResult

  return result
}

export async function answerClarifyBatch(params: {
  answers: Record<string, string>
  profile?: string
  questions: InboxRequestClarifyQuestion[]
  requestId: string
  request?: InboxRequest
}): Promise<ClarifyAnswerResult> {
  const { answers, profile, questions, requestId, request: rpc = defaultInboxRequest } = params
  let lastResult: ClarifyAnswerResult = { status: 'ok' }

  for (const q of questions) {
    const answer = answers[q.qid] ?? ''
    lastResult = await rpc('clarify.lock', {
      request_id: requestId,
      question_id: q.qid,
      answer,
      ...(profile ? { profile } : {})
    }) as ClarifyAnswerResult

    if (lastResult.status === 'expired') {break}
  }

  return lastResult
}

// ── refresh ──────────────────────────────────────────────────────────────────
export type InboxRequest = (method: string, params?: Record<string, unknown>) => Promise<unknown>

function defaultInboxRequest<Return = unknown>(method: string, params?: Record<string, unknown>): Promise<Return> {
  const gateway = $gateway.get()

  if (!gateway) {
    return Promise.reject(new Error('Inbox gateway is unavailable'))
  }

  return gateway.request(method, params ?? {}) as Promise<Return>
}

const ERROR_LIMIT = 240

function boundedError(error: unknown): string {
  const message =
    error instanceof Error
      ? error.message
      : isRecord(error) && typeof error.message === 'string'
        ? error.message
        : 'Inbox request failed'

  return message.trim().slice(0, ERROR_LIMIT) || 'Inbox request failed'
}

function isMethodNotFound(error: unknown): boolean {
  if (isRecord(error) && error.code === -32601) {return true}

  const message =
    error instanceof Error ? error.message : isRecord(error) && typeof error.message === 'string' ? error.message : ''

  return message.toLowerCase().includes('method not found') || message.toLowerCase().includes('method-not-found')
}

export interface InboxRefreshResult {
  published: boolean
  snapshot: InboxSnapshot | null
}

/**
 * Fetches the backend aggregation for the active connection + profile. A single
 * in-flight guard prevents overlapping requests; a gateway/profile switch bumps
 * the generation so a late response is dropped (A-B-A) instead of clobbering the
 * current selection.
 */
export async function refreshInbox(
  profile = '',
  request: InboxRequest = defaultInboxRequest
): Promise<InboxRefreshResult> {
  if ($inbox.get().capability === 'unsupported') {return { published: false, snapshot: null }}

  if (inboxRequestInFlight) {return { published: false, snapshot: null }}

  const token = beginInboxRequest(inboxRequestScope(profile))
  inboxRequestInFlight = token
  publishEntry({ ...$inbox.get(), loading: true })

  try {
    const response = await request('inbox.list', profile ? { profile } : {})

    if (!isInboxRequestCurrent(token)) {return { published: false, snapshot: null }}
    const snapshot = parseInboxSnapshot(response)

    if (!snapshot) {
      publishEntry({ ...$inbox.get(), error: 'Invalid inbox.list response', loading: false })

      return { published: false, snapshot: null }
    }

    const published = commitInboxRequest(token, snapshot)

    return { published, snapshot }
  } catch (error) {
    if (!isInboxRequestCurrent(token)) {return { published: false, snapshot: null }}

    if (isMethodNotFound(error)) {
      invalidateInboxRequests()
      publishEntry({ capability: 'unsupported', error: null, loading: false, snapshot: null })

      return { published: false, snapshot: null }
    }

    publishEntry({ ...$inbox.get(), error: boundedError(error), loading: false })

    return { published: false, snapshot: null }
  } finally {
    if (inboxRequestInFlight === token) {inboxRequestInFlight = null}
  }
}

// ── selectors ────────────────────────────────────────────────────────────────
export function selectInboxNeedsCount(entry: InboxEntry): number {
  return entry.snapshot?.counts.needs_you ?? 0
}

/** Case-insensitive substring search over title / session-key / cwd; empty query
 *  returns the input unchanged. Pure so the panel's search is unit-testable. */
export function filterInboxItems(items: InboxItem[], query: string): InboxItem[] {
  const needle = query.trim().toLowerCase()

  if (!needle) {return items}

  return items.filter(item => {
    const haystack = `${item.title}\n${item.session_key}\n${item.cwd}`.toLowerCase()

    return haystack.includes(needle)
  })
}

/** Mirrors the backend badge: amber when anyone needs the operator, red on a
 *  data-source error (never all-clear), muted otherwise. */
export function selectInboxBadge(entry: InboxEntry): InboxBadge {
  if (entry.capability === 'unsupported' || entry.error || entry.snapshot?.coverage.partial || (entry.snapshot?.coverage.errors.length ?? 0) > 0) {
    return 'red'
  }

  if (selectInboxNeedsCount(entry) > 0) {return 'amber'}

  return 'none'
}

// ── category filtering ────────────────────────────────────────────────────────

const RECOGNIZED_CATEGORIES = new Set(['goals', 'loops', 'heartbeats', 'subagents', 'background_tasks'])

export function filterByCategory(items: InboxItem[], category: InboxCategory): InboxItem[] {
  if (category === 'all') {return items}

  if (category === 'other') {
    return items.filter(item => !item.categories.some(c => RECOGNIZED_CATEGORIES.has(c)))
  }

  return items.filter(item => item.categories.includes(category))
}

/** Items with needs_you lane, used for the "Needs attention" nav lane. */
export function filterNeedsAttention(items: InboxItem[]): InboxItem[] {
  return items.filter(item => item.lanes.includes('needs_you'))
}

/** Count items in a category (for nav badges). */
export function countByCategory(items: InboxItem[], category: InboxCategory): number {
  return filterByCategory(items, category).length
}

/** Search scope: 'section' filters within category, 'all' bypasses category. */
export function searchInboxItems(
  items: InboxItem[],
  query: string,
  category: InboxCategory,
  scope: 'all' | 'section'
): InboxItem[] {
  const filtered = scope === 'all' ? items : filterByCategory(items, category)

  return filterInboxItems(filtered, query)
}
