/**
 * The create dialog's MCP setup target: WHICH connection and profile a New
 * Bot's credential setup writes to. Pure helpers below create-dialog.tsx —
 * the dialog owns the selection state, this module owns the identity rules.
 */

import type { McpSetupTarget } from './mcp-setup'
import type { ProfileRoute } from './types'

/** The create target's route: the picked connection's default backend door
 *  for remote targets, the active gateway (null) otherwise. Shared by the
 *  create dialog's request helper and its MCP setup target so both follow
 *  one selection. */
export function createTargetRoute(remoteTarget: boolean, connectionId: string): null | ProfileRoute {
  return remoteTarget
    ? {
        connectionId,
        mode: 'remote',
        profile: 'default',
        targetProfile: 'default'
      }
    : null
}

/** Complete setup target for the MCP buttons: the selection the flight was
 *  started under plus the materialized slug. A shared flight that created on
 *  ANOTHER selection retires instead of guessing a new owner. */
export function createdSetupTarget(args: {
  capturedSelection: string
  createdOn: string
  route: null | ProfileRoute
  slug: null | string
}): null | McpSetupTarget {
  return args.slug && args.createdOn === args.capturedSelection ? { route: args.route, profile: args.slug } : null
}
