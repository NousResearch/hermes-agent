// Runtime-info ownership for applyRuntimeInfo: the status-bar approval chip
// is keyed by the ACTIVE profile's name, so only the active gateway's runtime
// may reconcile it — the session.info event path applies the same rule.
// Another profile's `approvals.mode` written under the active name shows
// "Manual" on a profile that auto-approves, or the reverse.
import { reconcileApprovalModeForProfile } from '@/store/approval-mode'
import { activeGatewayConnectionId, isActivePrimary } from '@/store/gateway'
import { $activeGatewayProfile, normalizeProfileKey } from '@/store/profile'
import type { SessionOwnerRoute, SessionOwnerScope } from '@/store/session-request-router'
import type { SessionRuntimeInfo } from '@/types/hermes'

function ownerIsActiveGateway(owner: SessionOwnerScope): boolean {
  if (!owner) {
    return true
  }

  const activeProfile = normalizeProfileKey($activeGatewayProfile.get())

  if (typeof owner === 'string') {
    // A bare profile dials the profile door (the primary, or a connection-less
    // pool socket), so it is the active gateway only while that door is active,
    // never while a registry connection serving a same-named profile is.
    return normalizeProfileKey(owner) === activeProfile && (isActivePrimary() || activeGatewayConnectionId() === null)
  }

  return (
    normalizeProfileKey(owner.profile) === activeProfile &&
    (owner.connectionId?.trim() || 'local') === (activeGatewayConnectionId() ?? 'local')
  )
}

/** Reconcile `info`'s approval mode into the active profile's chip, but only
 *  when `owner` is the active gateway — a runtime another backend produced
 *  (a bot tile, a branch of a bot chat, an All-profiles resume) reports its
 *  OWN config and must not repaint the active profile's chip. */
export function reconcileOwnerApprovalMode(info: SessionRuntimeInfo, owner: SessionOwnerScope): void {
  if (info.approval_mode === undefined || !ownerIsActiveGateway(owner)) {
    return
  }

  reconcileApprovalModeForProfile($activeGatewayProfile.get(), info.approval_mode)
}

/** applyRuntimeInfo options for a runtime that must NOT publish into the main
 *  pane's composer (a session tile, a background branch): the owner is the
 *  create's route when the caller dialled one, else its profile. */
export function backgroundRuntimeOptions(
  route: null | SessionOwnerRoute | undefined,
  profile: null | string | undefined
): { foreground: false; owner: SessionOwnerScope } {
  return { foreground: false, owner: route ?? profile }
}
