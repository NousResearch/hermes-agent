import type { MutableRefObject } from 'react'

import type { ClientSessionState } from '@/app/types'
import { purgeInFlightTurnJournals } from '@/lib/inflight-turn-journal'
import { stampRowsWithOwningConnection } from '@/lib/session-owner-stamp'
import { clearQueuedPrompts } from '@/store/composer-queue'
import { resetSessionBackground } from '@/store/composer-status'
import { clearSessionGoal } from '@/store/goals'
import { $pinnedSessionIds } from '@/store/layout'
import { clearNotifications } from '@/store/notifications'
import { prunePreviewTabsForSession } from '@/store/preview'
import { $activeGatewayProfile } from '@/store/profile'
import { $projectTree, getProjectTreeConnectionId, refreshProjectTree } from '@/store/projects'
import {
  $connection,
  $cronSessions,
  $freshDraftKey,
  $messagingSessions,
  $sessions,
  $unlistedSessionOwnerRows,
  getSessionOwnerHint,
  getSessionOwnerHints,
  ownerLookupSessionRows,
  sessionMatchesStoredId,
  sessionPinId
} from '@/store/session'
import { clearSessionControl } from '@/store/session-control'
import { beginSessionMutation, tombstoneSessions, untombstoneSessions } from '@/store/session-removal'
import {
  profileScopeForSessionOwner,
  sessionOwnerRouteFromRow,
  type SessionOwnerScope
} from '@/store/session-request-router'
import {
  $sessionStates,
  $sessionTiles,
  closeSessionTile,
  dropSessionState,
  runtimeSessionOwner
} from '@/store/session-states'
import { sameSessionOwner } from '@/store/session-tile-identity'
import { forgetSessionUnread } from '@/store/session-unread'
import { $deleteKeepsNotifications, $sessionListSpansProfiles } from '@/store/shell-policy'
import { $archivedSessions } from '@/store/sidebar-archive'
import { clearSessionSubagents } from '@/store/subagents'
import { clearSessionTodos } from '@/store/todos'
import { dropTranscriptTail, dropTranscriptTailEverywhere } from '@/store/transcript-tail-cache'
import type { SessionInfo } from '@/types/hermes'

import { resumeRouteStillCurrent } from '../session-context-drift'

import {
  dropListedSession,
  findListedSession,
  type ListedSessionSlice,
  listedSliceTarget,
  restoreListedSession
} from './utils'

const listedStores = { sessions: $sessions, messaging: $messagingSessions, cron: $cronSessions }

/** Classic (main) dismisses every toast when a delete starts; see `$deleteKeepsNotifications`. */
export function clearNotificationsForDeletion(): void {
  if (!$deleteKeepsNotifications.get()) {
    clearNotifications()
  }
}

type ListedSession = { session: SessionInfo; slice: ListedSessionSlice } | undefined

interface DeletionCleanup {
  storedSessionId: string
  removedIds: Array<string | null | undefined>
  mutationIds: Array<string | null | undefined>
  closingRuntimeId: string | null
  mappedRuntimeId: string | undefined
  runtimeByStoredId: Map<string, string>
  statesByRuntimeId: Map<string, ClientSessionState>
}

interface SessionRemoval {
  removedIds: Array<string | null | undefined>
  mutationIds: Array<string | null | undefined>
  /** Puts the evicted rows back; returns the row whose usage a restored selection re-applies. */
  rollback: () => SessionInfo | undefined
}

export interface SessionDeletion {
  canClearId: (id: string) => boolean
  ownsRuntime: (runtimeId: string | null | undefined, storedId: string) => boolean
  removeRows: (storedSessionId: string, removed: SessionInfo | undefined) => SessionRemoval
  cleanup: (args: DeletionCleanup) => void
}

interface SessionDeletionPolicy {
  row: (storedId: string, explicit: SessionInfo | undefined, listed: SessionInfo | undefined) => SessionInfo | undefined
  route: (removed: SessionInfo | undefined, profile: string | undefined) => SessionOwnerScope
  owner: (
    owner: SessionOwnerScope,
    profile: string | undefined,
    listed: ListedSession,
    runtimeIdsForStoredId: (id: string) => Array<string | null | undefined>
  ) => SessionDeletion
  captureDraft: (
    getRouteToken: () => string,
    selectedId: MutableRefObject<string | null>,
    activeId: MutableRefObject<string | null>
  ) => () => boolean
}

/** A failed deletion may restore only the draft it opened, not newer navigation. */
function captureDeletionDraft(
  getRouteToken: () => string,
  selectedId: MutableRefObject<string | null>,
  activeId: MutableRefObject<string | null>
): () => boolean {
  const draftKey = $freshDraftKey.get()
  const routeToken = getRouteToken()
  const connection = $connection.get()
  const profile = $activeGatewayProfile.get()

  return () =>
    selectedId.current === null &&
    activeId.current === null &&
    $freshDraftKey.get() === draftKey &&
    resumeRouteStillCurrent(routeToken, getRouteToken(), '__new__') &&
    $connection.get() === connection &&
    $activeGatewayProfile.get() === profile
}

function rowOwner(row: SessionInfo): SessionOwnerScope {
  return sessionOwnerRouteFromRow(row) ?? (row.profile?.trim() || undefined)
}

function projectRows(): SessionInfo[] {
  return $projectTree
    .get()
    .flatMap(project => [
      ...(project.previewSessions ?? []),
      ...project.repos.flatMap(repo => repo.groups.flatMap(group => group.sessions))
    ])
}

/** Project rows retain profile-door shape; deletion needs their captured source. */
function ownerScopedDeletionRow(
  storedId: string,
  explicit: SessionInfo | undefined,
  listed: SessionInfo | undefined
): SessionInfo | undefined {
  const match = (row: SessionInfo) => sessionMatchesStoredId(row, storedId)
  const removed = explicit ?? listed ?? $archivedSessions.get().find(match)

  if (removed) {
    return removed
  }

  const project = projectRows().find(match)

  return project ? stampRowsWithOwningConnection([project], getProjectTreeConnectionId(), true)[0] : undefined
}

function ownerRows(): SessionInfo[] {
  return [
    ...ownerLookupSessionRows(),
    // The general lookup intentionally masks same-id stubs with visible rows.
    // Deletion must still see hidden canonical chats behind those rows.
    ...$unlistedSessionOwnerRows.get(),
    ...$archivedSessions.get(),
    ...stampRowsWithOwningConnection(projectRows(), getProjectTreeConnectionId(), true)
  ]
}

/** ID-only caches cannot distinguish twins. Preserve their shared slot whenever
 * another owner claims it; owner-tagged rows and transcript tails can be evicted
 * exactly. Re-read claims after awaits so a newly selected twin survives too. */
function ownerScopedDeletion(
  owner: SessionOwnerScope,
  runtimeIdsForStoredId: (id: string) => Array<string | null | undefined>
): SessionDeletion {
  const rowsAtStart = ownerRows()
  const tilesAtStart = $sessionTiles.get()
  const conflicts = (candidate: SessionOwnerScope) => Boolean(candidate && !sameSessionOwner(owner, candidate))

  const canClearId = (id: string): boolean => {
    if (getSessionOwnerHints(id).some(conflicts) || conflicts(runtimeSessionOwner(id))) {
      return false
    }

    if (runtimeIdsForStoredId(id).some(runtimeId => conflicts(runtimeSessionOwner(runtimeId)))) {
      return false
    }

    if ([...rowsAtStart, ...ownerRows()].some(row => sessionMatchesStoredId(row, id) && conflicts(rowOwner(row)))) {
      return false
    }

    if (
      [...tilesAtStart, ...$sessionTiles.get()].some(
        tile =>
          (tile.storedSessionId === id || tile.runtimeId === id) &&
          conflicts(tile.ownerRoute ?? tile.ownerProfile ?? runtimeSessionOwner(tile.runtimeId))
      )
    ) {
      return false
    }

    return !Object.entries($sessionStates.get()).some(
      ([runtimeId, state]) => state?.storedSessionId === id && conflicts(runtimeSessionOwner(runtimeId))
    )
  }

  const ownsRow = (row: SessionInfo): boolean => {
    const candidate = rowOwner(row)

    return candidate ? sameSessionOwner(owner, candidate) : canClearId(row.id)
  }

  const ownsRuntime = (runtimeId: string | null | undefined, storedId: string): boolean => {
    if (runtimeId && !canClearId(runtimeId)) {
      return false
    }

    const tile = $sessionTiles
      .get()
      .find(item => (runtimeId ? item.runtimeId === runtimeId : item.storedSessionId === storedId))

    const candidate =
      runtimeSessionOwner(runtimeId) ?? tile?.ownerRoute ?? tile?.ownerProfile ?? getSessionOwnerHint(storedId)

    return candidate ? sameSessionOwner(owner, candidate) : canClearId(storedId)
  }

  const removeRows = (storedSessionId: string, removed: SessionInfo | undefined): SessionRemoval => {
    const matchesRemoved = (row: SessionInfo) => sessionMatchesStoredId(row, storedSessionId) && ownsRow(row)
    const removedIds = [storedSessionId, removed?.id, removed?._lineage_root_id]
    const mutationIds = removedIds.filter((id): id is string => Boolean(id && canClearId(id)))
    const removedPinId = removed ? sessionPinId(removed) : storedSessionId
    const previousPins = $pinnedSessionIds.get()
    const removedPins = previousPins.filter(id => (id === storedSessionId || id === removedPinId) && canClearId(id))
    const stores = [$sessions, $messagingSessions, $cronSessions, $unlistedSessionOwnerRows, $archivedSessions]

    const snapshots = stores.map(store => {
      const rows = store.get()
      const removedRows = rows.filter(matchesRemoved)

      if (removedRows.length) {
        store.set(rows.filter(row => !matchesRemoved(row)))
      }

      return { store, removedRows }
    })

    tombstoneSessions(mutationIds)
    beginSessionMutation(mutationIds)
    $pinnedSessionIds.set(previousPins.filter(id => !removedPins.includes(id)))

    const rollback = () => {
      for (const { store, removedRows } of snapshots) {
        for (const row of removedRows) {
          const target =
            store === $unlistedSessionOwnerRows || store === $archivedSessions
              ? store
              : listedStores[listedSliceTarget(row)]

          const current = target.get()

          if (!current.some(existing => sessionMatchesStoredId(existing, row.id) && ownsRow(existing))) {
            target.set([row, ...current])
          }
        }
      }

      untombstoneSessions(mutationIds)
      $pinnedSessionIds.set([...new Set([...$pinnedSessionIds.get(), ...removedPins.filter(canClearId)])])

      // The listed lookup may now resolve a surviving twin; usage belongs to the deleted row.
      return removed
    }

    return { removedIds, mutationIds, rollback }
  }

  const cleanup = ({
    storedSessionId,
    removedIds,
    mutationIds,
    closingRuntimeId,
    mappedRuntimeId,
    runtimeByStoredId,
    statesByRuntimeId
  }: DeletionCleanup): void => {
    // With no twin claiming the id, sweep every scope (#94914: the save-path and delete-path
    // scopes derive from different sources); a twin keeps its own tail.
    if (canClearId(storedSessionId)) {
      dropTranscriptTailEverywhere(storedSessionId)
    } else if (owner) {
      dropTranscriptTail(storedSessionId, profileScopeForSessionOwner(owner))
    }

    const cleanupIds = removedIds.filter((id): id is string => Boolean(id && canClearId(id)))

    const cleanupRuntimeIds = [...new Set([closingRuntimeId, mappedRuntimeId])].filter((id): id is string =>
      Boolean(id && ownsRuntime(id, storedSessionId) && canClearId(id))
    )

    untombstoneSessions(mutationIds.filter(id => id && !canClearId(id)))

    if (cleanupIds.length !== removedIds.filter(Boolean).length) {
      // A global tombstone would hide the surviving twin. Refresh the tree
      // instead; its existing generation fence rejects stale source responses.
      void refreshProjectTree()
    }

    forgetSessionUnread(cleanupIds, typeof owner === 'string' ? owner : (owner?.targetProfile ?? owner?.profile))
    purgeInFlightTurnJournals([...cleanupIds, ...cleanupRuntimeIds])

    for (const id of cleanupIds) {
      clearQueuedPrompts(id)
      prunePreviewTabsForSession(id)
    }

    for (const id of cleanupRuntimeIds) {
      clearQueuedPrompts(id)
      clearSessionControl(id)
    }

    // Capture eligibility before closing a tile drops its runtime owner ledger.
    const tiledRuntimeId = mappedRuntimeId && cleanupRuntimeIds.includes(mappedRuntimeId) ? mappedRuntimeId : null
    const tiles = $sessionTiles.get().filter(tile => tile.storedSessionId === storedSessionId)

    if (
      tiles.length &&
      tiles.every(tile => {
        const tileOwner = tile.ownerRoute ?? tile.ownerProfile

        return tileOwner ? sameSessionOwner(owner, tileOwner) : ownsRuntime(tile.runtimeId, storedSessionId)
      })
    ) {
      closeSessionTile(storedSessionId)
    }

    if (tiledRuntimeId && runtimeByStoredId.get(storedSessionId) === tiledRuntimeId) {
      runtimeByStoredId.delete(storedSessionId)
      statesByRuntimeId.delete(tiledRuntimeId)
      dropSessionState(tiledRuntimeId)
    }

    for (const id of new Set([...cleanupIds, ...cleanupRuntimeIds])) {
      clearSessionSubagents(id)
      clearSessionTodos(id)
      clearSessionGoal(id)
      resetSessionBackground(id)
    }
  }

  return { canClearId, cleanup, ownsRuntime, removeRows }
}

/** Main's delete: the list shows one profile, so a stored id names one session and every
 *  id-keyed cache for it goes. */
function idScopedDeletion(profile: string | undefined, listed: ListedSession): SessionDeletion {
  const removeRows = (storedSessionId: string, removed: SessionInfo | undefined): SessionRemoval => {
    const previousPinned = $pinnedSessionIds.get()
    const previousArchived = $archivedSessions.get()
    // Pins are keyed on the durable lineage-root id; the stored id may be the
    // live tip after compression. Drop both so the pin can't linger.
    const removedPinId = removed ? sessionPinId(removed) : storedSessionId
    const removedIds = [storedSessionId, removed?.id, removed?._lineage_root_id]

    dropListedSession(storedSessionId)
    $archivedSessions.set(previousArchived.filter(session => !sessionMatchesStoredId(session, storedSessionId)))
    // Evict from the project tree's optimistic layer too (the backend snapshot
    // still lists it until its next refresh), so grouped + flat views drop the
    // row in lockstep. Pin the tombstone against the projects.tree prune while
    // the delete RPC is in flight, so a racing refresh can't flash it back.
    tombstoneSessions(removedIds)
    beginSessionMutation(removedIds)
    $pinnedSessionIds.set(previousPinned.filter(id => id !== storedSessionId && id !== removedPinId))

    const rollback = () => {
      if (listed) {
        restoreListedSession(listed.session, listed.slice)
      }

      // Restore the archived-view row too (no-op when it wasn't archived).
      $archivedSessions.set(previousArchived)

      untombstoneSessions(removedIds)
      $pinnedSessionIds.set(previousPinned)

      return findListedSession(storedSessionId)?.session
    }

    return { removedIds, mutationIds: removedIds, rollback }
  }

  const cleanup = ({
    storedSessionId,
    removedIds,
    closingRuntimeId,
    runtimeByStoredId,
    statesByRuntimeId
  }: DeletionCleanup): void => {
    dropTranscriptTailEverywhere(storedSessionId)
    // Only after the RPC lands — the optimistic eviction above can roll
    // back, and a rolled-back row must keep its watermark/marker.
    forgetSessionUnread(removedIds, profile)
    clearQueuedPrompts(storedSessionId)
    // The journaled in-flight tail holds this session's prompt and tool
    // calls in localStorage; a deleted session must not leave that copy
    // behind to age out on its own. Purge after the RPC lands (same
    // rollback argument as the unread watermark above), passing every id
    // the delete holds: the stored tip, the row id, the lineage root, and
    // the closing runtime id — the journal keys on the stored id.
    purgeInFlightTurnJournals([...removedIds, closingRuntimeId])

    // Preview tabs are session-owned: drop them with the session (pinned
    // tabs survive — they belong to the workspace, not the session).
    for (const id of removedIds) {
      if (id) {
        prunePreviewTabsForSession(id)
      }
    }

    if (closingRuntimeId) {
      clearQueuedPrompts(closingRuntimeId)
      clearSessionControl(closingRuntimeId)
    }

    // A tiled copy of this session must not outlive it: collapse the pane
    // and evict its mirrored runtime state so nothing submits to (or renders)
    // a deleted session.
    const tiledRuntimeId = runtimeByStoredId.get(storedSessionId)
    closeSessionTile(storedSessionId)

    if (tiledRuntimeId) {
      runtimeByStoredId.delete(storedSessionId)
      statesByRuntimeId.delete(tiledRuntimeId)
      dropSessionState(tiledRuntimeId)
    }

    // Live per-session stores (the same four the stop paths clear) key on
    // the gateway event's session_id, i.e. the runtime id. When the deleted
    // row is selected, closingRuntimeId is the foreground runtime, which can
    // differ from the stored→runtime mapping (cached/tiled runtime), so
    // clear the stored id and both runtime ids — once per distinct id.
    for (const sid of new Set([storedSessionId, closingRuntimeId, tiledRuntimeId].filter(Boolean) as string[])) {
      clearSessionSubagents(sid)
      clearSessionTodos(sid)
      clearSessionGoal(sid)
      resetSessionBackground(sid)
    }
  }

  return { canClearId: () => true, cleanup, ownsRuntime: () => true, removeRows }
}

const idScopedPolicy: SessionDeletionPolicy = {
  row: (storedId, _explicit, listed) =>
    listed ?? $archivedSessions.get().find(session => sessionMatchesStoredId(session, storedId)),
  route: (removed, profile) =>
    removed?.connection_id ? { connectionId: removed.connection_id, profile: removed.profile || 'default' } : profile,
  owner: (_owner, profile, listed) => idScopedDeletion(profile, listed),
  captureDraft: () => () => true
}

const ownerScopedPolicy: SessionDeletionPolicy = {
  row: ownerScopedDeletionRow,
  route: (removed, profile) => sessionOwnerRouteFromRow(removed) ?? profile,
  owner: (owner, _profile, _listed, runtimeIdsForStoredId) => ownerScopedDeletion(owner, runtimeIdsForStoredId),
  captureDraft: captureDeletionDraft
}

/** Read once per delete so a mid-delete shell flip cannot mix the two strategies. When the list
 *  spans profiles the same stored id can be twins in two owners, so only the deleted owner's
 *  rows and caches go; classic deletes by id exactly as main does. */
export function sessionDeletionPolicy(): SessionDeletionPolicy {
  return $sessionListSpansProfiles.get() ? ownerScopedPolicy : idScopedPolicy
}
