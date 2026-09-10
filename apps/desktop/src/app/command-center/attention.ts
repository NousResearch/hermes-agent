import type { SessionDotState } from '@/store/session-dot-state'
import type { SessionInfo } from '@/types/hermes'

export type AttentionKind = 'needs-input' | 'unread'
export type AttentionSource = 'messaging' | 'session'

export interface AttentionOwner {
  connectionId?: string
  profile: string
}

export type AttentionSession = Pick<
  SessionInfo,
  | '_lineage_root_id'
  | 'archived'
  | 'connection_id'
  | 'id'
  | 'last_active'
  | 'preview'
  | 'profile'
  | 'source'
  | 'started_at'
  | 'title'
  | 'unread'
>

export interface AttentionItem {
  connectionId?: string
  id: string
  kind: AttentionKind
  owner?: AttentionOwner
  profile: string
  sessionId: string
  source: AttentionSource
  timestamp: number
  title: string
}

export interface BuildAttentionItemsOptions {
  attentionOwners?: Readonly<Record<string, readonly AttentionOwner[]>>
  attentionSessionIds: readonly string[]
  dotStates: Readonly<Record<string, SessionDotState>>
  messagingSessions: readonly AttentionSession[]
  sessions: readonly AttentionSession[]
}

const ATTENTION_PRIORITY: Record<AttentionKind, number> = {
  'needs-input': 0,
  unread: 1
}

const profileKey = (row: AttentionSession): string => row.profile?.trim() || 'default'

const connectionKey = (row: AttentionSession): string => row.connection_id?.trim() || 'local'

const lineageId = (row: AttentionSession): string => row._lineage_root_id?.trim() || row.id

const matchesSessionId = (row: AttentionSession, id: string): boolean => row.id === id || row._lineage_root_id === id

const ownerMatches = (row: AttentionSession, owner: AttentionOwner): boolean =>
  profileKey(row) === owner.profile && connectionKey(row) === (owner.connectionId?.trim() || 'local')

const isWaitingRow = (
  row: AttentionSession,
  id: string,
  rows: readonly { row: AttentionSession }[],
  attentionOwners: Readonly<Record<string, readonly AttentionOwner[]>> | undefined
): boolean => {
  const hintedOwners = Object.entries(attentionOwners ?? {})
    .filter(([candidateId]) => matchesSessionId(row, candidateId))
    .flatMap(([, owners]) => owners)

  if (hintedOwners.length > 0) {
    return hintedOwners.some(owner => ownerMatches(row, owner))
  }

  const matchingRows = rows.filter(({ row: candidate }) => !candidate.archived && matchesSessionId(candidate, id))

  const matchingOwners = new Set(
    matchingRows.map(({ row: candidate }) => `${profileKey(candidate)}\u0000${connectionKey(candidate)}`)
  )

  return matchingOwners.size === 1
}

const rowTitle = (row: AttentionSession): string => row.title?.trim() || row.preview?.trim() || `Session ${row.id}`

const rowTimestamp = (row: AttentionSession): number => row.last_active || row.started_at || 0

const itemKey = (row: AttentionSession): string =>
  `${profileKey(row)}\u0000${row.connection_id?.trim() || 'local'}\u0000${lineageId(row)}`

const itemFromRow = (row: AttentionSession, kind: AttentionKind, source: AttentionSource): AttentionItem => ({
  connectionId: row.connection_id?.trim() || undefined,
  id: itemKey(row),
  kind,
  owner: {
    connectionId: row.connection_id?.trim() || undefined,
    profile: profileKey(row)
  },
  profile: profileKey(row),
  sessionId: row.id,
  source,
  timestamp: rowTimestamp(row),
  title: rowTitle(row)
})

const syntheticWaitingItem = (sessionId: string): AttentionItem => ({
  id: `default\u0000local\u0000${sessionId}`,
  kind: 'needs-input',
  owner: undefined,
  profile: 'default',
  sessionId,
  source: 'session',
  timestamp: 0,
  title: `Session ${sessionId}`
})

/**
 * Build the persistent Desktop attention list from existing session state.
 *
 * This function deliberately owns no state and performs no navigation. It is
 * the single precedence/deduplication policy used by the Command Center and
 * its titlebar count, keeping those surfaces from drifting apart.
 */
export function buildAttentionItems({
  attentionOwners,
  attentionSessionIds,
  dotStates,
  messagingSessions,
  sessions
}: BuildAttentionItemsOptions): AttentionItem[] {
  const rows: Array<{ row: AttentionSession; source: AttentionSource }> = [
    ...sessions.map(row => ({ row, source: 'session' as const })),
    ...messagingSessions.map(row => ({ row, source: 'messaging' as const }))
  ]

  const waitingIds = new Set(attentionSessionIds)
  const byKey = new Map<string, AttentionItem>()

  const add = (item: AttentionItem): void => {
    const existing = byKey.get(item.id)

    if (!existing || ATTENTION_PRIORITY[item.kind] < ATTENTION_PRIORITY[existing.kind]) {
      byKey.set(item.id, item)
    }
  }

  for (const { row, source } of rows) {
    if (row.archived) {
      continue
    }

    const kind = [...waitingIds].some(id => matchesSessionId(row, id) && isWaitingRow(row, id, rows, attentionOwners))
      ? 'needs-input'
      : dotStates[row.id] === 'unread' || row.unread === true
        ? 'unread'
        : null

    if (kind) {
      add(itemFromRow(row, kind, source))
    }
  }

  for (const sessionId of attentionSessionIds) {
    const hasVisibleRow = rows.some(({ row }) => matchesSessionId(row, sessionId) && !row.archived)
    const hasArchivedRow = rows.some(({ row }) => matchesSessionId(row, sessionId) && row.archived)

    if (!hasVisibleRow && !hasArchivedRow) {
      add(syntheticWaitingItem(sessionId))
    }
  }

  return [...byKey.values()].sort((left, right) => {
    const priority = ATTENTION_PRIORITY[left.kind] - ATTENTION_PRIORITY[right.kind]

    if (priority !== 0) {
      return priority
    }

    return right.timestamp - left.timestamp || left.title.localeCompare(right.title)
  })
}
