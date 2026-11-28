import { $connectionsRegistry } from '@/store/connections'
import { $focusedWorkspaceCwd } from '@/store/session-states'
import { focusedWorkspaceExecutionOwner, workspaceExecutionOwnerMatchesActiveBackend } from '@/store/workspace-execution-owner'

import type { TerminalGateway } from './terminal-gateway'
import { shellGatewayFor } from './terminal-gateway-routing'

/** Resolve the focused folder together with its shell host. URL/cloud gateways
 * have no shell transport: preserve upstream's local shell, without sending a
 * remote project path to it. An explicit host/folder choice bypasses this default. */
export function focusedTerminalWorkspace(fallback: TerminalGateway | undefined): { cwd: string; gateway: TerminalGateway | undefined } {
  const owner = focusedWorkspaceExecutionOwner()
  const registry = $connectionsRegistry.get()
  const gateway = owner ? shellGatewayFor(owner.connectionId, owner.profile, registry) : undefined

  if (gateway) {
    const sameHost = owner?.mode === 'local' || gateway.connectionId === owner?.connectionId

    return { cwd: sameHost ? $focusedWorkspaceCwd.get() : '', gateway }
  }

  // Legacy single-gateway hosts have no registry; only use a folder whose
  // owner is the active backend. Unknown/removed owners never donate paths.
  return {
    cwd: !registry && workspaceExecutionOwnerMatchesActiveBackend(owner) ? $focusedWorkspaceCwd.get() : '',
    gateway: fallback
  }
}
