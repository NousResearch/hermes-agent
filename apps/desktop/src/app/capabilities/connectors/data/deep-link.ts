// The browser leg of an ACCOUNT connect came back through
// `hermes://connections/done`.
//
// A chat's operation lands on the tool row that opened it
// (`openConnectionDoneLink`). An account operation has no session and no tool
// row, so it lands on the Connectors page instead, with the app's dialog open.
//
// Nothing in the link is trusted to move a row: the op id only names which
// operation to show, and the backend reads the account itself. The wake only
// shortens the wait.

import { CAPABILITIES_ROUTE } from '../../../routes'

import { $accountOperations } from './account-operations'
import { wakeAccountOperation } from './rpc'

/** The Connectors page URL that opens one app's dialog. */
export const connectorRoute = (slug: string): string =>
  `${CAPABILITIES_ROUTE}?tab=connectors&connector=${encodeURIComponent(slug)}`

/**
 * Handle the returning link when THIS window holds the account operation it
 * names. Returns false when it does not, so the caller can fall through to the
 * session path.
 *
 * An operation that already settled is ignored: the tab can come back long after
 * the person moved on, and a stale link must not pull them away from where they
 * are.
 */
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
    // The operation can settle and leave the live registry between the link and
    // this RPC (4004); the watcher reads the account at its next tick anyway.
  }

  return true
}
