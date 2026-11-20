/**
 * `ctx.confirmAction` — let a plugin's local service learn that the user
 * deliberately confirmed ONE exact proposal (#158148).
 *
 * The renderer only forwards: Electron main shows a native dialog the page
 * cannot click, and signs a receipt only when the user presses Confirm
 * (electron/plugin-confirmation.ts). Cancel, expiry, a busy dialog or a bad
 * proposal never produce a receipt. The plugin hands the receipt to its
 * service, which verifies the signature against the key Desktop publishes at
 * `$HERMES_HOME/desktop/action-confirmation-key.json` and checks every bound
 * field (action, revision, profile, service instance, nonce) against what it
 * proposed before it acts.
 *
 * Closed unless the plugin declares `capabilities: ['confirm.action']`.
 * Resolves a result and never throws.
 *
 * The gate holds because the raw door is not reachable from plugin code: the
 * preload hands it out once, and this module claims it while it is evaluated,
 * which the runtime loader's static import puts before any plugin is. A plugin
 * sharing the realm can only call the closure its own ctx carries, with its
 * own id and its own declared capabilities.
 */

export type {
  HermesActionConfirmation as PluginActionConfirmation,
  HermesActionProposal as PluginActionProposal,
  HermesSignedActionReceipt as PluginSignedActionReceipt
} from '@/global'

import type { HermesActionConfirmation, HermesActionProposal, HermesConfirmPluginAction } from '@/global'

/** The capability a plugin declares to use `ctx.confirmAction`. */
export const CONFIRM_ACTION_CAPABILITY = 'confirm.action'

// A hot reload re-evaluates this module; it keeps the door it already holds,
// since the preload hands it out only once.
const claimed: HermesConfirmPluginAction | null =
  (import.meta.hot?.data?.confirmBridge as HermesConfirmPluginAction | undefined) ??
  (typeof window === 'undefined' ? null : (window.hermesDesktop?.claimPluginConfirmation?.() ?? null))

if (import.meta.hot?.data) {
  import.meta.hot.data.confirmBridge = claimed
}

export function createPluginConfirmAction(
  pluginId: string,
  capabilities: readonly string[] = []
): (proposal: HermesActionProposal) => Promise<HermesActionConfirmation> {
  const declared = capabilities.includes(CONFIRM_ACTION_CAPABILITY)

  return async proposal => {
    if (!declared) {
      return {
        status: 'refused',
        error: `Plugin "${pluginId}" did not declare the ${CONFIRM_ACTION_CAPABILITY} capability.`
      }
    }

    if (!claimed) {
      return { status: 'unavailable', error: 'This Hermes Desktop build cannot confirm plugin actions.' }
    }

    try {
      return await claimed(pluginId, proposal)
    } catch (error) {
      return { status: 'unavailable', error: error instanceof Error ? error.message : String(error) }
    }
  }
}
