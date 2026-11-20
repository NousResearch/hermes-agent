import type { WorkspaceMode } from '@/contrib/types'

import { normalizeProfileKey } from './profile'
import { TILE_PANE_PREFIX } from './session-focus'
import { isSessionOwnerRoute, type SessionOwnerScope } from './session-request-router'
import type { SessionTile } from './session-states'

export function ownerProfileKey(owner: SessionOwnerScope): string | undefined {
  if (isSessionOwnerRoute(owner)) {
    return normalizeProfileKey(owner.profile)
  }

  return typeof owner === 'string' ? normalizeProfileKey(owner) : undefined
}

export function sameSessionOwner(left: SessionOwnerScope, right: SessionOwnerScope): boolean {
  if (isSessionOwnerRoute(left) && isSessionOwnerRoute(right)) {
    return (
      left.connectionId.trim() === right.connectionId.trim() &&
      normalizeProfileKey(left.profile) === normalizeProfileKey(right.profile) &&
      normalizeProfileKey(left.targetProfile ?? left.profile) ===
        normalizeProfileKey(right.targetProfile ?? right.profile)
    )
  }

  if (typeof left === 'string' && typeof right === 'string') {
    return normalizeProfileKey(left) === normalizeProfileKey(right)
  }

  return false
}

export function tileWorkspaceMode(tile: SessionTile): WorkspaceMode {
  return tile.workspaceMode ?? 'sessions'
}

export function tilesShareOwner(left: SessionTile, right: SessionTile): boolean {
  if (left.workspaceMode && right.workspaceMode && left.workspaceMode !== right.workspaceMode) {
    return false
  }

  const workspaceMode = left.workspaceMode ?? right.workspaceMode ?? 'sessions'

  if (workspaceMode === 'bots') {
    if (left.workspaceOwnerKey && right.workspaceOwnerKey) {
      return left.workspaceOwnerKey === right.workspaceOwnerKey
    }

    if (left.ownerRoute && right.ownerRoute) {
      return sameSessionOwner(left.ownerRoute, right.ownerRoute)
    }

    return true
  }

  if (left.ownerRoute && right.ownerRoute) {
    return sameSessionOwner(left.ownerRoute, right.ownerRoute)
  }

  return true
}

export function mergeSessionTile(previous: SessionTile, next: SessionTile, storedSessionId: string): SessionTile {
  const merged: SessionTile = { ...previous, storedSessionId }

  if (next.anchor !== undefined) {
    merged.anchor = next.anchor
  }

  if (next.before !== undefined) {
    merged.before = next.before
  }

  if (next.dir !== undefined) {
    merged.dir = next.dir
  }

  if (next.error !== undefined) {
    merged.error = next.error
  }

  if (next.ownerRoute !== undefined) {
    merged.ownerRoute = next.ownerRoute
  }

  if (next.runtimeId !== undefined) {
    merged.runtimeId = next.runtimeId
  }

  if (next.workspaceMode !== undefined) {
    merged.workspaceMode = next.workspaceMode
  }

  if (next.workspaceOwnerKey !== undefined) {
    merged.workspaceOwnerKey = next.workspaceOwnerKey
  }

  if (next.workspaceTabTitle !== undefined) {
    merged.workspaceTabTitle = next.workspaceTabTitle
  }

  return merged
}

/** Persisted placement — `dir` + strip slot (`before`) + dock `anchor` so a
 *  restart / profile swap re-adopts tiles in the same order, not all stacked
 *  right of workspace. */
export type StoredTile = Pick<
  SessionTile,
  | 'anchor'
  | 'before'
  | 'dir'
  | 'ownerProfile'
  | 'ownerRoute'
  | 'storedSessionId'
  | 'workspaceMode'
  | 'workspaceOwnerKey'
  | 'workspaceTabTitle'
>

export const toStoredTile = (t: SessionTile): StoredTile => ({
  anchor: t.anchor,
  before: t.before,
  dir: t.dir,
  ...(t.ownerProfile ? { ownerProfile: t.ownerProfile } : {}),
  ...(t.ownerRoute ? { ownerRoute: t.ownerRoute } : {}),
  storedSessionId: t.storedSessionId,
  ...(t.workspaceMode ? { workspaceMode: t.workspaceMode } : {}),
  ...(t.workspaceOwnerKey ? { workspaceOwnerKey: t.workspaceOwnerKey } : {}),
  ...(t.workspaceTabTitle ? { workspaceTabTitle: t.workspaceTabTitle } : {})
})

export function parseTileList(value: unknown): StoredTile[] {
  return Array.isArray(value)
    ? value
        .filter((t): t is SessionTile => Boolean(t && typeof (t as SessionTile).storedSessionId === 'string'))
        .map(t => {
          const raw = t as SessionTile

          return {
            // #108679: a tile whose anchor is its OWN pane id is
            // self-referential — the re-dock target can never exist (the
            // pane is not in the tree at adoption time), so the dock falls
            // through to an arbitrary same-placement neighbor instead of the
            // recorded layout. Rewrite it to the workspace anchor at load,
            // the same surface an anchorless tile re-docks against.
            anchor:
              typeof raw.anchor === 'string' && raw.anchor !== `${TILE_PANE_PREFIX}${raw.storedSessionId}`
                ? raw.anchor
                : undefined,
            before: typeof raw.before === 'string' || raw.before === null ? raw.before : undefined,
            dir: raw.dir,
            ownerProfile: typeof raw.ownerProfile === 'string' ? normalizeProfileKey(raw.ownerProfile) : undefined,
            ownerRoute:
              raw.ownerRoute &&
              typeof raw.ownerRoute.connectionId === 'string' &&
              typeof raw.ownerRoute.profile === 'string'
                ? {
                    connectionId: raw.ownerRoute.connectionId,
                    mode: raw.ownerRoute.mode,
                    profile: raw.ownerRoute.profile,
                    ...(typeof raw.ownerRoute.targetProfile === 'string'
                      ? { targetProfile: raw.ownerRoute.targetProfile }
                      : {})
                  }
                : undefined,
            storedSessionId: raw.storedSessionId,
            workspaceMode: raw.workspaceMode === 'bots' ? 'bots' : 'sessions',
            workspaceOwnerKey:
              raw.workspaceMode === 'bots' && typeof raw.workspaceOwnerKey === 'string'
                ? raw.workspaceOwnerKey
                : undefined,
            workspaceTabTitle: typeof raw.workspaceTabTitle === 'string' ? raw.workspaceTabTitle : undefined
          }
        })
    : []
}
