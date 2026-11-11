import type { MutableRefObject } from 'react'

import type { ClientSessionState } from '@/app/types'
import { purgeInFlightTurnJournals } from '@/lib/inflight-turn-journal'
import { stampRowsWithOwningConnection } from '@/lib/session-owner-stamp'
import { clearQueuedPrompts } from '@/store/composer-queue'
import { resetSessionBackground } from '@/store/composer-status'
import { clearSessionGoal } from '@/store/goals'
import { $pinnedSessionIds } from '@/store/layout'
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
import { $archivedSessions } from '@/store/sidebar-archive'
import { clearSessionSubagents } from '@/store/subagents'
import { clearSessionTodos } from '@/store/todos'
import { dropTranscriptTail } from '@/store/transcript-tail-cache'
import type { SessionInfo } from '@/types/hermes'

import { resumeRouteStillCurrent } from '../session-context-drift'

import { listedSliceTarget } from './utils'

const listedStores = { sessions: $sessions, messaging: $messagingSessions, cron: $cronSessions }

/** A failed deletion may restore only the draft it opened, not newer navigation. */
export function captureDeletionDraft(
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

/** Classic project rows retain profile-door shape; deletion needs their captured source. */
export function sessionDeletionRow(
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
export function sessionDeletionOwner(
  owner: SessionOwnerScope,
  runtimeIdsForStoredId: (id: string) => Array<string | null | undefined>
) {
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

  const removeRows = (storedSessionId: string, removed: SessionInfo | undefined) => {
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
  }: {
    storedSessionId: string
    removedIds: Array<string | null | undefined>
    mutationIds: string[]
    closingRuntimeId: string | null
    mappedRuntimeId: string | undefined
    runtimeByStoredId: Map<string, string>
    statesByRuntimeId: Map<string, ClientSessionState>
  }): void => {
    if (owner || canClearId(storedSessionId)) {
      dropTranscriptTail(storedSessionId, profileScopeForSessionOwner(owner))
    }

    const cleanupIds = removedIds.filter((id): id is string => Boolean(id && canClearId(id)))

    const cleanupRuntimeIds = [...new Set([closingRuntimeId, mappedRuntimeId])].filter((id): id is string =>
      Boolean(id && ownsRuntime(id, storedSessionId) && canClearId(id))
    )

    untombstoneSessions(mutationIds.filter(id => !canClearId(id)))

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
