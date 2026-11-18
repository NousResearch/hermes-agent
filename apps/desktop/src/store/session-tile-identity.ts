import type { WorkspaceMode } from '@/contrib/types'

import { normalizeProfileKey } from './profile'
import { isSessionOwnerRoute, type SessionOwnerScope } from './session-request-router'
import type { SessionTile } from './session-states'

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
