import { normalizeProfileKey } from './profile'
import { isSessionOwnerRoute, type SessionOwnerScope } from './session-request-router'

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
