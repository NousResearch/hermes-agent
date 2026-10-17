import type { ReloadMcpResult } from '@hermes/shared'

import type { HermesGateway } from '@/hermes'
import { translateNow } from '@/i18n'
import { $gateway } from '@/store/gateway'
import { notify } from '@/store/notifications'

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

/** The change has not reached the chat yet; the user already has the deferral notice. */
export class McpReloadDeferred extends Error {}

export async function reloadMcp(
  gateway: Pick<HermesGateway, 'request'>,
  sessionId: null | string | undefined
): Promise<McpReloadOutcome> {
  const result = await gateway.request<ReloadMcpResult>('reload.mcp', {
    confirm: true,
    session_id: sessionId ?? undefined
  })

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

/** For a flow that claims `server` is live in the chat (connect, reconnect): resolves only once it is;
 *  rejects with {@link McpReloadDeferred} while a turn defers it, or an error when it did not connect. */
export async function reloadMcpFor(server: string, sessionId: null | string | undefined): Promise<void> {
  const gateway = $gateway.get()

  if (!gateway) {
    throw new Error(translateNow('settings.mcp.reloadFailed'))
  }

  const outcome = await reloadMcp(gateway, sessionId)
  announceMcpReload(outcome, [server])

  if (outcome.unavailable.includes(server)) {
    throw new Error(translateNow('promptNotices.mcpReloadUnavailable', server))
  }

  if (outcome.deferred) {
    throw new McpReloadDeferred(translateNow('promptNotices.mcpReloadDeferred'))
  }

  // Connected for the profile is not enabled in this chat (no_mcp, an explicit pick, a platform allowlist).
  if (outcome.sessionServers && !outcome.sessionServers.includes(server)) {
    throw new Error(translateNow('promptNotices.mcpReloadNotInChat', server))
  }
}
