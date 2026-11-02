import { type ProfileScope, profileScopeKey } from '@/hermes'
import { activeGatewayConnectionId } from '@/store/gateway'

import { CAPABILITIES_ROUTE } from '../../../routes'

import { $accountOperations } from './account-operations'
import { wakeAccountOperation } from './rpc'

const connectorRoute = (slug: string): string =>
  `${CAPABILITIES_ROUTE}?tab=connectors&connector=${encodeURIComponent(slug)}`

/** The Connectors tab, scoped to the chat's own profile and connection, with one app's dialog and that
 *  account's rename editor open. The account goes by id: its name can change and be reused by another. */
export function accountRenameRoute(
  slug: string,
  connectionId: string,
  owner: { connectionId: null | string; profile: string }
) {
  const params = new URLSearchParams({ profile: owner.profile, rename: connectionId })

  if (owner.connectionId) {
    params.set('connection', owner.connectionId)
  }

  return `${connectorRoute(slug)}&${params.toString()}`
}

export async function resumeAccountConnect(opId: string, navigate: (to: string) => void): Promise<boolean> {
  const operation = $accountOperations.get()[opId]

  if (!operation) {
    return false
  }

  if (operation.settled) {
    return true
  }

  navigate(operation.returnRoute ?? connectorRoute(operation.connectors[0] ?? ''))

  try {
    await wakeAccountOperation(operation.scope, opId)
  } catch {
    // The operation can settle and leave the live registry between the link and this RPC.
  }

  return true
}

/** The scope a link names: the profile, pinned to its connection when that is not the active one. */
export function linkedScope(params: URLSearchParams): null | string | { connectionId: string; profile: string } {
  const profile = params.get('profile')?.trim()

  if (!profile) {
    return null
  }

  const connectionId = params.get('connection')?.trim()

  // The active connection is the ambient route; only another connection needs a pinned scope.
  return connectionId && connectionId !== (activeGatewayConnectionId() ?? 'local') ? { connectionId, profile } : profile
}

const claimedLinks = new Set<string>()

/** Whether this mounted Connectors tab handles the link at `locationKey`. A tab for another scope waits for
 *  the page to switch to the link's scope, and of several mounted pages only the first takes it. */
export function claimRouteLink(locationKey: string, params: URLSearchParams, scope: ProfileScope): boolean {
  const linked = linkedScope(params)

  if (linked !== null && profileScopeKey(linked) !== profileScopeKey(scope)) {
    return false
  }

  const claim = `${locationKey}?${params.toString()}`

  if (claimedLinks.has(claim)) {
    return false
  }

  claimedLinks.add(claim)

  return true
}

export function resetRouteLinksForTests(): void {
  claimedLinks.clear()
}
