import { backendScopeKey, LOCAL_CONNECTION_ID } from '@hermes/shared'

import { Codecs, persistentAtom } from '@/lib/persisted'
import type { SessionInfo } from '@/types/hermes'

import { $activeGatewayProfile } from './profile'
import { $projectTree } from './project-tree'
import {
  $connection,
  $cronSessions,
  $messagingSessions,
  $sessions,
  sessionMatchesStoredId,
  sessionPinId
} from './session'
import type { SessionDotState } from './session-dot-state'

/** Presentation only: never acknowledges messages or changes execution/approval policy.
 * Buckets use the registry's (connection, profile) identity; entries use durable lineage ids.
 * Unmuting reveals unread activity accumulated while muted. This is client-local, not a server setting. */
export const $sessionUnreadMutes = persistentAtom<Record<string, string[]>>(
  'hermes.desktop.sessionUnreadMutes',
  {},
  Codecs.json(value => {
    if (!value || typeof value !== 'object' || Array.isArray(value)) {
      return {}
    }

    return Object.fromEntries(
      Object.entries(value).flatMap(([scope, ids]) =>
        Array.isArray(ids)
          ? [[scope, [...new Set(ids.filter((id): id is string => typeof id === 'string' && !!id))]]]
          : []
      )
    )
  })
)

export type UnreadMuteRow = Pick<SessionInfo, 'id' | 'profile' | 'connection_id' | '_lineage_root_id' | '_lineage_ids'>

export const unreadMuteScope = (row: Pick<SessionInfo, 'profile' | 'connection_id'>): string =>
  backendScopeKey(row.connection_id, row.profile)

export const scopedSessionDotKey = (row: UnreadMuteRow): string => JSON.stringify([unreadMuteScope(row), row.id])

export function isSessionUnreadMuted(row: UnreadMuteRow): boolean {
  return $sessionUnreadMutes.get()[unreadMuteScope(row)]?.includes(sessionPinId(row)) ?? false
}

export function setSessionUnreadMuted(row: UnreadMuteRow, muted: boolean): void {
  const scope = unreadMuteScope(row)
  const id = sessionPinId(row)
  const current = $sessionUnreadMutes.get()
  const ids = current[scope] ?? []

  if (ids.includes(id) === muted) {
    return
  }

  const next = { ...current }
  const kept = muted ? [...ids, id] : ids.filter(candidate => candidate !== id)

  if (kept.length) {
    next[scope] = kept
  } else {
    delete next[scope]
  }

  $sessionUnreadMutes.set(next)
}

/** Mask only unread, preserving live/attention priority and every other owner's same-id row. */
export function applyUnreadMuteOverrides(dots: Record<string, SessionDotState>, rows: readonly UnreadMuteRow[]): void {
  const mutedAliases = new Map<string, boolean>()

  for (const row of rows) {
    const muted = isSessionUnreadMuted(row)

    if (muted && dots[row.id] === 'unread') {
      dots[scopedSessionDotKey(row)] = 'idle'
    }

    // No cross-profile alias expansion here: each row declares its own lineage.
    for (const alias of [row.id, row._lineage_root_id, ...(row._lineage_ids ?? [])]) {
      if (alias) {
        mutedAliases.set(alias, (mutedAliases.get(alias) ?? true) && muted)
      }
    }
  }

  for (const id of Object.keys(dots)) {
    if (dots[id] !== 'unread') {
      continue
    }

    if (mutedAliases.get(id)) {
      delete dots[id]
    }
  }
}

/** Menus can belong to project-only rows or hidden chats, not just recents. */
export function unreadMuteRowFor(storedId: string, profile?: string): UnreadMuteRow {
  const owner = profile || $activeGatewayProfile.get()

  const rows = [
    ...$sessions.get(),
    ...$cronSessions.get(),
    ...$messagingSessions.get(),
    ...$projectTree
      .get()
      .flatMap(project => [
        ...(project.previewSessions ?? []),
        ...project.repos.flatMap(repo => repo.groups.flatMap(group => group.sessions))
      ])
  ]

  return (
    rows.find(row => sessionMatchesStoredId(row, storedId) && (row.profile || 'default') === (owner || 'default')) ?? {
      id: storedId,
      profile: owner,
      connection_id: $connection.get()?.connectionId || LOCAL_CONNECTION_ID
    }
  )
}

export function dropUnreadMutesForScope(connectionId: null | string | undefined, profile: string): void {
  const current = $sessionUnreadMutes.get()
  const scope = backendScopeKey(connectionId, profile)

  if (scope in current) {
    const next = { ...current }
    delete next[scope]
    $sessionUnreadMutes.set(next)
  }
}

export function migrateUnreadMutesForProfile(from: string, to: string): void {
  const current = $sessionUnreadMutes.get()
  const oldScope = backendScopeKey(LOCAL_CONNECTION_ID, from)
  const newScope = backendScopeKey(LOCAL_CONNECTION_ID, to)

  if (oldScope === newScope || !current[oldScope]) {
    return
  }

  const next = { ...current, [newScope]: [...new Set([...(current[newScope] ?? []), ...current[oldScope]])] }
  delete next[oldScope]
  $sessionUnreadMutes.set(next)
}
