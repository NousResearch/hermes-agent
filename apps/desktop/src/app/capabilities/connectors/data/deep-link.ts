import { CAPABILITIES_ROUTE } from '../../../routes'

import { $accountOperations } from './account-operations'
import { wakeAccountOperation } from './rpc'

const connectorRoute = (slug: string): string =>
  `${CAPABILITIES_ROUTE}?tab=connectors&connector=${encodeURIComponent(slug)}`

/** The Connectors tab, scoped to the chat's own profile and connection, with one app's dialog and the named
 *  account's rename editor open. */
export function accountRenameRoute(
  slug: string,
  alias: string,
  owner: { connectionId: null | string; profile: string }
) {
  const params = new URLSearchParams({ profile: owner.profile, rename: alias })

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

  navigate(connectorRoute(operation.connectors[0] ?? ''))

  try {
    await wakeAccountOperation(operation.scope, opId)
  } catch {
    // The operation can settle and leave the live registry between the link and this RPC.
  }

  return true
}
