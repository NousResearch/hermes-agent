import type { ReloadMcpResult } from '@hermes/shared'

import type { HermesGateway } from '@/hermes'
import { translateNow } from '@/i18n'
import { $gateway } from '@/store/gateway'
import { notify } from '@/store/notifications'

/** A profile-wide reconcile can outlast the default RPC budget on a machine with many servers. */
export const MCP_RELOAD_TIMEOUT_MS = 150_000

/** What the calling chat's committed policy did: applied, already current, or deferred by a turn. */
export type McpReloadSession = 'applied' | 'unchanged' | 'busy'

/**
 * What `reload.mcp` did for the chat that asked. The gateway decides and commits; Desktop only reads its
 * answer, so a successful RPC envelope alone never means the chat got the change.
 */
export interface McpReloadOutcome {
  /** A turn was running or queued: the chat keeps its servers until the turn finishes and MCP reloads
   *  again, or until it is reopened. */
  deferred: boolean
  /** The calling chat's policy generation, or null when the call had no session (or a non-local one). */
  session: McpReloadSession | null
  /** Configured servers that did not connect, beside whatever did apply. */
  unavailable: string[]
  /** The MCP servers the chat's committed policy runs; null from a gateway that does not report them. */
  sessionServers: null | string[]
}

/** The reply fields every `reload.mcp` caller reads; older gateways may omit each of them. */
export interface McpReloadReply {
  message: null | string
  session: McpReloadSession | null
  status: null | string
  unavailable: string[]
}

/** The change has not reached the chat yet; the user already has the deferral notice. */
export class McpReloadDeferred extends Error {}

const SESSION_STATES = new Set<McpReloadSession>(['applied', 'unchanged', 'busy'])

const sessionState = (value: unknown): McpReloadSession | null =>
  typeof value === 'string' && SESSION_STATES.has(value as McpReloadSession) ? (value as McpReloadSession) : null

const names = (value: unknown): string[] =>
  Array.isArray(value) ? value.filter((name): name is string => typeof name === 'string') : []

/** Read a `reload.mcp` reply once: the request path and the slash renderer share the same defaults. */
export function parseMcpReloadReply(response: unknown): McpReloadReply {
  const record = response && typeof response === 'object' ? (response as Record<string, unknown>) : {}

  return {
    message: typeof record.message === 'string' ? record.message : null,
    session: sessionState(record.session),
    status: typeof record.status === 'string' ? record.status : null,
    // `failed` is the pre-unification name an older gateway still answers.
    unavailable: Array.isArray(record.unavailable) ? names(record.unavailable) : names(record.failed)
  }
}

/** `now`/`approve`/`once`/`yes` skip the consent gate; `always` also persists the profile's opt-out. */
const MCP_RELOAD_CONFIRM_WORDS = new Set(['always', 'approve', 'now', 'once', 'yes'])

/** Arguments the Desktop's `/reload-mcp` forwards: a bare command still asks the backend for consent. */
export function mcpReloadSlashParams(arg: string, sessionId: null | string | undefined): Record<string, unknown> {
  const choice = arg.trim().toLowerCase()

  return {
    ...(sessionId ? { session_id: sessionId } : {}),
    ...(MCP_RELOAD_CONFIRM_WORDS.has(choice) ? { confirm: true } : {}),
    ...(choice === 'always' ? { always: true } : {})
  }
}

/** The stable prefix an older gateway uses to reject `profile` at admission (code 4000, before the
 *  handler runs). Such a gateway reloads every profile, so retrying once without it still reaches the
 *  target profile as well as that backend can. */
const MCP_RELOAD_PROFILE_REJECTION = 'invalid params for reload.mcp: profile:'

async function requestMcpReload(
  gateway: Pick<HermesGateway, 'request'>,
  params: Record<string, unknown>
): Promise<ReloadMcpResult> {
  try {
    return await gateway.request<ReloadMcpResult>('reload.mcp', params, MCP_RELOAD_TIMEOUT_MS)
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error)

    if (!('profile' in params) || !message.includes(MCP_RELOAD_PROFILE_REJECTION)) {
      throw error
    }

    const { profile: _dropped, ...withoutProfile } = params

    return gateway.request<ReloadMcpResult>('reload.mcp', withoutProfile, MCP_RELOAD_TIMEOUT_MS)
  }
}

export async function reloadMcp(
  gateway: Pick<HermesGateway, 'request'>,
  sessionId: null | string | undefined,
  profile?: null | string
): Promise<McpReloadOutcome> {
  const result = await requestMcpReload(gateway, {
    confirm: true,
    ...(profile ? { profile } : {}),
    session_id: sessionId ?? undefined
  })

  const reply = parseMcpReloadReply(result)

  if (reply.status === 'confirm_required') {
    throw new Error(reply.message ?? translateNow('connectorsPage.reload.confirmRequired'))
  }

  return {
    deferred: reply.session === 'busy',
    session: reply.session,
    sessionServers: result?.session_servers ?? null,
    unavailable: reply.unavailable
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
