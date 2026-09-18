import { atom } from 'nanostores'

import { getApiRequestConnection } from '@/hermes'

import { $gateway } from './gateway'

/**
 * Agent Inbox — a read-only aggregation + navigation surface over the ACTIVE
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
  cwd: string
  heartbeat: Record<string, unknown> | null
  goal: Record<string, unknown> | null
  lanes: InboxLane[]
  loop: Record<string, unknown> | null
  pending_approval: InboxPendingApproval | null
  pending_clarify: InboxPendingClarify | null
  session_key: string
  source: string
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

  return {
    cwd: asString(value.cwd),
    goal: value.goal === null ? null : isRecord(value.goal) ? value.goal : null,
    heartbeat: value.heartbeat === null ? null : isRecord(value.heartbeat) ? value.heartbeat : null,
    lanes: value.lanes as InboxLane[],
    loop: value.loop === null ? null : isRecord(value.loop) ? value.loop : null,
    pending_approval: pendingApproval,
    pending_clarify: pendingClarify,
    session_key: asString(value.session_key),
    source: asString(value.source),
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
