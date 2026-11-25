import { hermesApi } from '@/api/client'
import { getSession, mutateSessionHttp } from '@/api/sessions'
import { LOCAL_SESSION_SOURCE_IDS, normalizeSessionSource } from '@/lib/session-source'
import { DAY, formatDate, formatDateTime, relativeTime } from '@/lib/time'
import { $disbandedGatewayRooms, canonicalGroupKey } from '@/plugins/hermes-bots/canonical-group-registry'
import { type CanonicalRoom, listCanonicalRooms } from '@/plugins/hermes-bots/canonical-groups'
import { $groupChats } from '@/plugins/hermes-bots/group-chat'
import { gatewayGroupEngine } from '@/plugins/hermes-bots/group-engine'
import type { GroupChat, GroupMember } from '@/plugins/hermes-bots/types'
import { requestGatewayForAgent } from '@/store/gateway'
import type { PaginatedSessions, SessionInfo } from '@/types/hermes'

import type { AgentRouteOwner, AgentSettingsOwner } from './data'

export const HISTORY_PAGE_SIZE = 25
export type HistorySource = 'all' | 'chat' | 'schedule' | 'telegram' | 'discord'
export interface HistoryFilters {
  source: Array<Exclude<HistorySource, 'all'>>
  archived: 'exclude' | 'include' | 'only'
  group: 'project' | 'source' | 'none'
}
export const DEFAULT_HISTORY_FILTERS: HistoryFilters = { source: [], archived: 'exclude', group: 'project' }
export interface CleanupCriteria {
  days: 0 | 7 | 30 | 90
  messages: 'any' | 'under5' | 'empty'
  source: HistorySource
  includeArchived: boolean
}
export const DEFAULT_CLEANUP: CleanupCriteria = { days: 30, messages: 'any', source: 'all', includeArchived: false }

const CHAT_SOURCES = [
  ...LOCAL_SESSION_SOURCE_IDS.filter(source => !['kanban', 'oneshot'].includes(source)),
  'gui',
  'web'
]

const SOURCE_IDS: Record<Exclude<HistorySource, 'all'>, string[]> = {
  chat: CHAT_SOURCES,
  schedule: ['cron'],
  telegram: ['telegram'],
  discord: ['discord']
}

export function historySource(row: SessionInfo): string {
  const source = normalizeSessionSource(row.source) ?? ''

  if (CHAT_SOURCES.includes(source)) {
    return 'chat'
  }

  return source === 'cron' ? 'schedule' : source
}

export function visibleHistory(row: SessionInfo): boolean {
  // Canonical chats remain private even if an older runtime has not run its hide sweep.
  return !row.hidden && row.title !== 'Bot Chat' && !row.is_internal_child
}

export function matchesHistory(row: SessionInfo, filters: HistoryFilters, search = ''): boolean {
  return (
    visibleHistory(row) &&
    (filters.archived === 'include' || Boolean(row.archived) === (filters.archived === 'only')) &&
    (!filters.source.length || filters.source.some(source => historySource(row) === source)) &&
    `${row.title ?? ''}\n${row.preview ?? ''}\n${row.id}`
      .toLocaleLowerCase()
      .includes(search.trim().toLocaleLowerCase())
  )
}

export function historyRows(pages: PaginatedSessions[]): SessionInfo[] {
  const rows = new Map<string, SessionInfo>()

  for (const page of pages) {
    for (const row of page.sessions) {
      rows.set(row._lineage_root_id || row.id, row)
    }
  }

  return [...rows.values()].filter(visibleHistory)
}

function requireOwner(owner: AgentSettingsOwner): asserts owner is AgentSettingsOwner & { connectionId: string } {
  if (!owner.connectionId || !owner.profile) {
    throw new Error('History requires an explicit connection and agent')
  }
}

/** The facts every listed room carries: its name, its seats and its last change. */
interface AgentGroupChatFacts {
  name: string
  /** The room's seats, counted from the membership its own surfaces read. */
  seats: number
  /** Seconds the room last changed; 0 when its store keeps no clock. */
  lastActive: number
}

/** One group chat the agent is seated in, as History lists it. Rooms are not sessions — the
 *  sessions a room drives are hidden `bot_room` plumbing — so History lists the rooms themselves:
 *  a canonical gateway room (bound, then opened by its key) or a Desktop room record (by name). */
export type AgentGroupChat =
  | (AgentGroupChatFacts & { kind: 'gateway'; room: CanonicalRoom })
  | (AgentGroupChatFacts & { kind: 'local' })

/** Whether a stored member descriptor is this owner: members are keyed `connectionId::profile`
 *  (`botRosterKey`), and the unsourced descriptors legacy rooms hold are this machine's own
 *  (`groupMemberKey`), so a bare name can never seat another machine's agent. */
function memberSeatsOwner(member: GroupMember | null | undefined, owner: AgentRouteOwner): boolean {
  if (member?.name !== owner.profile) {
    return false
  }

  const connection = member.route?.connectionId ?? member.connectionId

  return connection ? connection === owner.connectionId : !member.remoteSource && owner.connectionId === 'local'
}

/** The Desktop rooms that seat this owner, minus the ones the gateway already listed (one row a
 *  room). A room the local store records as this agent's gateway room counts even while its members
 *  are unread; a disband tombstone and every other agent's room do not. */
function localGroupChats(
  rooms: Record<string, GroupChat>,
  owner: AgentRouteOwner,
  listed: Set<string>
): AgentGroupChat[] {
  return Object.entries(rooms).flatMap(([name, room]) => {
    if (!room || room.tombstone) {
      return []
    }

    const engine = gatewayGroupEngine(room).engine
    const owned = Boolean(engine && engine.connectionId === owner.connectionId && engine.profile === owner.profile)
    const members = room.members ?? []
    const listedRoom = listed.has(room.roomId ?? engine?.roomId ?? '')

    if (listedRoom || (!owned && !members.some(member => memberSeatsOwner(member, owner)))) {
      return []
    }

    return [
      {
        kind: 'local' as const,
        name,
        seats: members.length,
        lastActive: room.log.reduce((newest, entry) => Math.max(newest, entry.at || 0), 0) / 1000
      }
    ]
  })
}

/** The agent's group chats, newest first: the rooms its own route holds, plus the Desktop rooms the
 *  local store records for it. Other agents' rooms, disbanded rooms and empty name placeholders are
 *  never among them. */
export function agentGroupChats(
  owner: AgentRouteOwner,
  rooms: Record<string, GroupChat>,
  gatewayRooms: readonly CanonicalRoom[]
): AgentGroupChat[] {
  // A disbanded room can still arrive in a `groups.list` that started before the disband: the room's
  // own tombstone and this window's disband set both retire it, as the Bots roster does.
  const disbanded = $disbandedGatewayRooms.get()
  const gateway: AgentGroupChat[] = gatewayRooms
    .filter(room => !room.disbanded_at && !disbanded.has(canonicalGroupKey(owner, room.room_id)))
    .map(room => ({
      kind: 'gateway',
      name: room.name,
      seats: room.members?.length ?? 0,
      lastActive: room.updated_at ?? 0,
      room
    }))

  return [...gateway, ...localGroupChats(rooms, owner, new Set(gatewayRooms.map(room => room.room_id)))].sort(
    (a, b) => b.lastActive - a.lastActive || a.name.localeCompare(b.name)
  )
}

/** A gateway that predates group chats has none, and a room list that fails must not take the
 *  agent's conversations down with it: the rooms that answered are listed, the local store's always. */
async function routeGroupChats(owner: AgentRouteOwner): Promise<CanonicalRoom[]> {
  try {
    return await listCanonicalRooms(owner)
  } catch {
    return []
  }
}

/** The agent's group chats for the History tab. Read-only: History opens a room, never mutates one. */
export async function getAgentGroupChats(owner: AgentSettingsOwner): Promise<AgentGroupChat[]> {
  requireOwner(owner)

  return agentGroupChats(owner, $groupChats.get(), await routeGroupChats(owner))
}

export async function getHistoryPage(
  owner: AgentSettingsOwner,
  filters: HistoryFilters,
  offset = 0
): Promise<PaginatedSessions> {
  requireOwner(owner)

  const query = new URLSearchParams({
    profile: owner.profile,
    limit: String(HISTORY_PAGE_SIZE),
    offset: String(offset),
    archived: filters.archived,
    order: 'recent',
    min_messages: '0'
  })

  if (filters.source.length) {
    query.set('sources', filters.source.flatMap(source => SOURCE_IDS[source]).join(','))
  }

  const page = await hermesApi<PaginatedSessions>({
    ...owner,
    path: `/api/profiles/sessions?${query}`,
    timeoutMs: 60_000
  })

  if (page.errors?.length || Object.keys(page.storage ?? {}).length) {
    throw new Error(page.errors?.map(error => error.error).join('; ') || 'Session storage unavailable')
  }

  if (!Number.isFinite(page.total) || page.offset !== offset || page.limit <= 0) {
    throw new Error('History pagination unavailable; update Hermes and retry')
  }

  return page
}

export function nextHistoryOffset(page: PaginatedSessions): number | undefined {
  const next = page.offset + page.limit

  return next < page.total ? next : undefined
}

interface LiveHistoryRow {
  id?: string
  session_key?: string
  status?: string
}

export async function getHistoryRunning(owner: AgentSettingsOwner): Promise<Set<string>> {
  requireOwner(owner)

  const response = await requestGatewayForAgent<{ sessions: LiveHistoryRow[] }>(
    owner.connectionId,
    owner.profile,
    'session.active_list',
    { profile: owner.profile }
  )

  if (!Array.isArray(response.sessions)) {
    throw new Error('Running session status unavailable')
  }

  return new Set(
    response.sessions
      .filter(row => row.status !== 'idle')
      .flatMap(row => [row.id, row.session_key].filter((id): id is string => Boolean(id)))
  )
}

const HOUR_S = 3_600
const DAY_S = 86_400

/** The table's "Last active": recent activity reads relative ("4 min. ago", "3 hr. ago",
 *  "Yesterday"), older activity a date ("Oct 30, 2026"); the caller shows `absolute` on hover. */
export function historyLastActive(
  seconds: number,
  locale: string,
  nowMs = Date.now()
): { label: string; absolute: string } {
  const date = new Date(seconds * 1000)
  const absolute = formatDateTime(date, locale)
  const age = Math.max(0, nowMs / 1000 - seconds)
  const today = new Date(nowMs)
  today.setHours(0, 0, 0, 0)
  const days = Math.ceil((today.getTime() / 1000 - seconds) / DAY_S)

  // The previous calendar day takes the shared formatter's own day word ("yesterday"); anything
  // older takes the shared date. Both come from lib/time, so every redesign surface reads alike.
  const label =
    age < HOUR_S || days <= 0
      ? relativeTime(seconds * 1000, nowMs, locale)
      : days === 1
        ? relativeTime(nowMs - DAY, nowMs, locale)
        : formatDate(date)

  return { label: label.charAt(0).toLocaleUpperCase(locale) + label.slice(1), absolute }
}

/** A row's line under its title in confirmations: "Chat · Nov 1, 2026 · 0 messages". */
export function historyRowDate(row: SessionInfo, locale: string): string {
  const seconds = row.last_active || row.started_at

  return seconds ? formatDate(new Date(seconds * 1000), locale) : ''
}

export function historyRunning(row: SessionInfo, running: Set<string>): boolean {
  return (
    row.scheduler_owned === true ||
    [row.id, row._lineage_root_id, ...(row._lineage_ids ?? [])].some(id => Boolean(id && running.has(id)))
  )
}

export function cleanupMatches(row: SessionInfo, criteria: CleanupCriteria, now: number): boolean {
  const lastActive = row.last_active || row.started_at

  return (
    matchesHistory(row, {
      source: criteria.source === 'all' ? [] : [criteria.source],
      archived: criteria.includeArchived ? 'include' : 'exclude',
      group: 'none'
    }) &&
    (criteria.days === 0 || (lastActive > 0 && lastActive < now - criteria.days * 86_400)) &&
    (criteria.messages === 'any' || row.message_count < (criteria.messages === 'empty' ? 1 : 5))
  )
}

export interface CleanupPreview {
  candidates: SessionInfo[]
  running: SessionInfo[]
  criteria: CleanupCriteria
  now: number
}

export async function previewHistoryCleanup(
  owner: AgentSettingsOwner,
  criteria: CleanupCriteria,
  current: () => boolean
): Promise<CleanupPreview> {
  const pages: PaginatedSessions[] = []
  let offset: number | undefined = 0

  // The legacy prune endpoint excludes ALL open rows, not just running work. Use the
  // ordinary complete inventory and canonical per-session deletes instead of that bulk writer.
  while (offset !== undefined) {
    if (!current()) {
      throw new Error('History owner changed')
    }

    const page = await getHistoryPage(
      owner,
      {
        source: criteria.source === 'all' ? [] : [criteria.source],
        archived: criteria.includeArchived ? 'include' : 'exclude',
        group: 'none'
      },
      offset
    )

    pages.push(page)
    offset = nextHistoryOffset(page)
  }

  const running = await getHistoryRunning(owner)

  if (!current()) {
    throw new Error('History owner changed')
  }

  const now = Date.now() / 1000
  const rows = historyRows(pages).filter(row => cleanupMatches(row, criteria, now))

  return {
    candidates: rows.filter(row => !historyRunning(row, running)),
    running: rows.filter(row => historyRunning(row, running)),
    criteria: { ...criteria },
    now
  }
}

class HistoryRowChanged extends Error {}

export async function mutateHistorySession(
  owner: AgentSettingsOwner,
  id: string,
  method: 'PATCH' | 'DELETE',
  payload: Record<string, unknown>,
  current: () => boolean,
  upgradeMessage: string,
  validate?: (row: SessionInfo) => void
): Promise<{ ok: boolean; title?: string; already_absent?: boolean }> {
  requireOwner(owner)

  if (!current()) {
    throw new Error('History owner changed')
  }

  return mutateSessionHttp(
    id,
    method,
    payload,
    owner,
    identity => {
      if (!current()) {
        throw new Error('History owner changed')
      }

      // Deletion must be generation-fenced and run through the authority's transactional
      // require_idle check. Never fall back to an older unfenced bulk/DELETE endpoint.
      if (!identity.request_id) {
        throw new Error(upgradeMessage)
      }
    },
    async () => {
      if (!current()) {
        throw new Error('History owner changed')
      }

      const row = await getSession(id, owner)

      if (!visibleHistory(row)) {
        throw new HistoryRowChanged('Session is now protected')
      }

      validate?.(row)
    }
  )
}

export interface HistoryDeleteOutcome {
  deleted: string[]
  running: string[]
  changed: string[]
  errors: Array<{ id: string; error: string }>
}

export async function deleteHistorySelection(
  owner: AgentSettingsOwner,
  rows: SessionInfo[],
  current: () => boolean,
  upgradeMessage: string,
  cleanup?: CleanupPreview
): Promise<HistoryDeleteOutcome> {
  const result: HistoryDeleteOutcome = { deleted: [], running: [], changed: [], errors: [] }
  const running = await getHistoryRunning(owner)

  for (const selected of rows) {
    if (!current()) {
      break
    }

    try {
      // Read AFTER the mutation snapshot, so any edit after this validation
      // conflicts with the captured revision instead of widening consent.
      const response = await mutateHistorySession(owner, selected.id, 'DELETE', {}, current, upgradeMessage, row => {
        if (cleanup && !cleanupMatches(row, cleanup.criteria, cleanup.now)) {
          throw new HistoryRowChanged('Session no longer matches cleanup')
        }

        if (historyRunning(row, running)) {
          throw new Error('session_busy')
        }
      })

      if (!response.ok) {
        throw new Error('Deletion was not acknowledged')
      }

      if (response.already_absent) {
        result.changed.push(selected.id)
      } else {
        result.deleted.push(selected.id)
      }
    } catch (error) {
      if (!current()) {
        break
      }

      const message = error instanceof Error ? error.message : String(error)
      let becameRunning = false

      if (/revision_conflict|stale_generation/.test(message)) {
        try {
          becameRunning = historyRunning(selected, await getHistoryRunning(owner))
        } catch {
          /* The original refusal remains actionable; never retry a destructive write. */
        }
      }

      if (error instanceof HistoryRowChanged) {
        result.changed.push(selected.id)
      } else if (becameRunning || /session_busy|unknown_execution|cannot delete an active session/.test(message)) {
        result.running.push(selected.id)
      } else {
        result.errors.push({ id: selected.id, error: message })
      }
    }
  }

  return result
}
