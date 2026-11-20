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
 */

export type {
  HermesActionConfirmation as PluginActionConfirmation,
  HermesActionProposal as PluginActionProposal,
  HermesSignedActionReceipt as PluginSignedActionReceipt
} from '@/global'

import type { HermesActionConfirmation, HermesActionProposal } from '@/global'

/** The capability a plugin declares to use `ctx.confirmAction`. */
export const CONFIRM_ACTION_CAPABILITY = 'confirm.action'

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

    const bridge = typeof window === 'undefined' ? undefined : window.hermesDesktop?.confirmPluginAction

    if (!bridge) {
      return { status: 'unavailable', error: 'This Hermes Desktop build cannot confirm plugin actions.' }
    }

    try {
      return await bridge(pluginId, proposal)
    } catch (error) {
      return { status: 'unavailable', error: error instanceof Error ? error.message : String(error) }
    }
  }
}
