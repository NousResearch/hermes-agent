import type { ReloadMcpResult } from '@hermes/shared'

import { translateNow } from '@/i18n'
import { notify } from '@/store/notifications'
import { ambientRequestFor } from '@/store/session-gone-latch'
import { requestForOwnedSession } from '@/store/session-states'
import { refreshToolMediaOwners } from '@/store/tool-media-owners'

/**
 * What `reload.mcp` did for the chat that asked. The gateway decides and commits; Desktop only reads its
 * answer, so a successful RPC envelope alone never means the chat got the change.
 */
export interface McpReloadOutcome {
  /** A turn was running or queued: the chat keeps its servers until the turn finishes and MCP reloads
   *  again, or until it is reopened. */
  deferred: boolean
  /** Configured servers that did not connect, beside whatever did apply. */
  unavailable: string[]
  /** The MCP servers the chat's committed policy runs; null from a gateway that does not report them. */
  sessionServers: null | string[]
}

export async function reloadMcp(
  gateway: Parameters<typeof ambientRequestFor>[0],
  sessionId: null | string | undefined
): Promise<McpReloadOutcome> {
  // The chat's MCP set lives on its owner; the ambient socket refuses (or
  // reloads the wrong profile for) another profile's session.
  const result = await requestForOwnedSession<ReloadMcpResult>(sessionId, ambientRequestFor(gateway), 'reload.mcp', {
    confirm: true,
    session_id: sessionId ?? undefined
  })

  // Reconnected servers can register tools a package's renderer was waiting on.
  refreshToolMediaOwners()

  return {
    deferred: result?.session === 'busy',
    sessionServers: result?.session_servers ?? null,
    unavailable: result?.unavailable ?? []
  }
}

/** Tell the user what did not take effect: a deferred chat, and servers (other than `skip`) that did not
 *  connect. */
export function announceMcpReload(outcome: McpReloadOutcome, skip: readonly string[] = []): void {
  if (outcome.deferred) {
    notify({ id: 'mcp-reload-deferred', kind: 'info', message: translateNow('promptNotices.mcpReloadDeferred') })
  }

  const unavailable = outcome.unavailable.filter(name => !skip.includes(name))

  if (unavailable.length > 0) {
    notify({
      id: 'mcp-reload-unavailable',
      kind: 'warning',
      message: translateNow('promptNotices.mcpReloadUnavailable', unavailable.join(', ')),
      title: translateNow('settings.mcp.reloadFailed')
    })
  }
}
