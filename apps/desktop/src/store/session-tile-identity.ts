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
