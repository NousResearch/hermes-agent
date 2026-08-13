import { authMcpServer, cancelMcpOAuthFlow, getMcpOAuthFlow, listMcpServers } from '@/hermes'
import { translateNow } from '@/i18n'
import { completeMcpDesktopOAuth, McpOAuthCancelled } from '@/lib/mcp-dashboard-oauth'
import { prettyName } from '@/lib/text'
import { type ComposerSuggestion, offerSuggestions } from '@/store/composer-suggestions'
import { $gateway } from '@/store/gateway'
import { notifyError } from '@/store/notifications'

/**
 * Connection-repair event provider: an MCP tool call just failed with an
 * auth/connection error, so offer to reconnect that server right where the
 * user is about to type their next message. The highest-precision source on
 * the bus — state-triggered by an actual failure in THIS session, never
 * guessed from keywords.
 *
 * Fed by the gateway event stream (`reportMcpToolResult` from the
 * tool.complete handler). The offer self-limits: a successful reconnect —
 * or any later successful call to the same server — withdraws it.
 */

// Auth/connection shaped failure text from an MCP tool result. Deliberately
// narrow: a tool erroring on its own semantics ("issue not found") must not
// pill a reconnect. Matched against the error/result string.
const REPAIR_RE = /\b(401|403|unauthorized|forbidden|token .*(expired|invalid)|oauth|authenticat\w+ (failed|required|expired)|connection (refused|closed|reset|failed)|server (unavailable|disconnected|not connected)|ECONNREFUSED)\b/i

/** `mcp__figma__get_design_context` → `figma`; null for non-MCP tools. */
export const mcpServerFromToolName = (toolName: string): string | null => {
  const match = /^mcp__([^_]+(?:_[^_]+)*?)__/.exec(toolName)

  return match ? match[1]! : null
}

const failed = new Map<string, Set<string>>()

const keyFor = (sessionId: string | null | undefined): string => sessionId ?? ''

async function reconnect(server: string, sessionId: string | null, cancelled: () => boolean): Promise<void> {
  try {
    await completeMcpDesktopOAuth({
      serverName: server,
      start: authMcpServer,
      status: getMcpOAuthFlow,
      cancelled,
      cancel: cancelMcpOAuthFlow,
      openExternal: url => window.hermesDesktop.openExternal(url)
    })

    // Fresh tokens reach the live session before the pill claims success.
    await $gateway
      .get()
      ?.request('reload.mcp', { confirm: true, session_id: sessionId ?? undefined })
      .catch(() => {})

    clearMcpFailure(sessionId, server)
  } catch (error) {
    if (!(error instanceof McpOAuthCancelled)) {
      notifyError(error, translateNow('composer.repairSuggestions.failed', prettyName(server)))
    }

    throw error
  }
}

function toSuggestion(server: string, sessionId: string | null): ComposerSuggestion {
  const copy = (key: string, ...args: unknown[]) => translateNow(`composer.repairSuggestions.${key}`, ...args)
  const name = prettyName(server)

  return {
    brand: server,
    doneLabel: copy('done', name),
    doneTip: copy('doneTip'),
    icon: 'plug',
    id: server,
    invoke: context => reconnect(server, sessionId, context.cancelled),
    label: copy('label', name),
    provider: 'repair',
    tip: copy('tip', name),
    workingLabel: copy('working', name),
    workingTip: copy('workingTip')
  }
}

function publish(sessionId: string | null | undefined): void {
  const servers = [...(failed.get(keyFor(sessionId)) ?? [])]

  offerSuggestions(
    sessionId,
    'repair',
    servers.map(server => toSuggestion(server, sessionId ?? null))
  )
}

/**
 * Called from the gateway tool.complete handler for every finished tool call.
 * A failed MCP call with auth/connection-shaped output raises the offer for
 * its server; a successful call to that server withdraws it (the connection
 * evidently works again).
 */
export function reportMcpToolResult(
  sessionId: string | null | undefined,
  toolName: string,
  isError: boolean,
  resultText: string
): void {
  const server = mcpServerFromToolName(toolName)

  if (!server) {
    return
  }

  const key = keyFor(sessionId)
  const sessionFailed = failed.get(key)

  if (isError && REPAIR_RE.test(resultText)) {
    // Only offer repair for servers that are actually configured — a failure
    // from a just-removed server shouldn't pitch reconnecting it.
    void listMcpServers()
      .then(({ servers }) => {
        if (!servers.some(entry => entry.name === server)) {
          return
        }

        const next = failed.get(key) ?? new Set<string>()

        next.add(server)
        failed.set(key, next)
        publish(sessionId)
      })
      .catch(() => {
        // Can't verify the server exists — don't offer.
      })
  } else if (!isError && sessionFailed?.delete(server)) {
    publish(sessionId)
  }
}

/** Withdraw a server's repair offer (reconnected, or server removed). */
export function clearMcpFailure(sessionId: string | null | undefined, server: string): void {
  if (failed.get(keyFor(sessionId))?.delete(server)) {
    publish(sessionId)
  }
}
