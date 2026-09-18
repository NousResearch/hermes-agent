// What a server on this Mac is doing right now, and what it costs.
//
// Extracted from `mcp-tab.tsx`: the Connectors page derives a local card's state
// from exactly this `ServerStatus`, so the two surfaces can never disagree about
// whether a server is off, probing, broken or answering.

import { compactNumber } from '@hermes/shared'

import { getUsageAnalytics, type McpTestResult, type ProfileScope } from '@/hermes'
import type { Translations } from '@/i18n'
import { estimateServerTokens, serverUsageCount } from '@/lib/mcp-cost'
import { NEEDS_AUTH_RE } from '@/lib/mcp-probe-cache'
import { countEnabledTools } from '@/lib/mcp-tool-filter'

import { serverEnabled } from './mcp-doc'

// Shared cache for the Nous-approved catalog — feeds both description enrichment
// and the Catalog install view; invalidated after an install.
export const MCP_CATALOG_KEY = ['mcp-catalog'] as const

export type Probe = McpTestResult | 'probing'

export type ServerStatus = 'error' | 'needs-auth' | 'off' | 'ok' | 'probing' | 'unknown'

// Per-server cost/usage overlay inputs: `tokens` is the approximate per-call
// schema cost from the probe (null = no estimate — older backend or no probe
// yet), `uses` is the 30-day analytics call count (null = analytics
// unavailable, so usage is simply omitted).
export interface ServerCost {
  tokens: null | number
  uses: null | number
}

// 30-day per-tool call counts for the MCP fleet — same shape and TTL rules as
// the Toolsets tab's toolCallsCache, but a 30-day window keyed by the
// Capabilities scope profile. Purely cosmetic: a failed analytics fetch caches
// nothing and the overlay omits usage.
export const MCP_USAGE_TTL_MS = 10 * 60_000
const mcpUsageCache = new Map<string, { at: number; value: Record<string, number> }>()

export async function loadMcpUsage(
  scopeKey: string,
  scopeProfile: ProfileScope
): Promise<null | Record<string, number>> {
  const cached = mcpUsageCache.get(scopeKey)

  if (cached && Date.now() - cached.at < MCP_USAGE_TTL_MS) {
    return cached.value
  }

  try {
    const analytics = await getUsageAnalytics(30, scopeProfile)
    const value = Object.fromEntries((analytics.tools ?? []).map(entry => [entry.tool, entry.count]))
    mcpUsageCache.set(scopeKey, { at: Date.now(), value })

    return value
  } catch {
    // Analytics unavailable — degrade to "no usage shown", never an error UI.
    return null
  }
}

export function statusOf(server: Record<string, unknown>, probe: Probe | undefined): ServerStatus {
  if (!serverEnabled(server)) {
    return 'off'
  }

  if (probe === 'probing') {
    return 'probing'
  }

  if (!probe) {
    return 'unknown'
  }

  if (probe.ok) {
    return 'ok'
  }

  return NEEDS_AUTH_RE.test(probe.error ?? '') ? 'needs-auth' : 'error'
}

export const STATUS_DOT: Record<ServerStatus, string> = {
  ok: 'bg-emerald-500',
  error: 'bg-red-500',
  'needs-auth': 'bg-amber-500',
  probing: 'animate-pulse bg-foreground/40',
  off: 'bg-foreground/20',
  unknown: 'bg-foreground/20'
}

/** Tools the agent actually registers for one server, given its include/exclude
 *  filter. `null` until a successful probe says what the server has. */
export function enabledToolCount(server: Record<string, unknown>, probe: Probe | undefined): null | number {
  if (!probe || probe === 'probing' || !probe.ok) {
    return null
  }

  return countEnabledTools(
    server,
    probe.tools.map(tool => tool.name)
  )
}

// "12 tools enabled" / "25 tools, 1 prompts, 103 resources enabled" — only
// the capabilities the server actually has. When a `server` config is passed,
// the tool count reflects the per-tool include/exclude filter (what's actually
// registered), not the raw discovered count. The optional `cost` appends the
// overlay — "…, ~4.2k tok, 3 uses/30d" — with each half omitted when unknown.
export function capabilitySummary(
  m: Translations['settings']['mcp'],
  probe: McpTestResult,
  server?: Record<string, unknown>,
  cost?: ServerCost
): string {
  const toolCount = server
    ? countEnabledTools(
        server,
        probe.tools.map(tool => tool.name)
      )
    : probe.tools.length

  const parts = [m.capabilitySummary(toolCount, probe.prompts ?? 0, probe.resources ?? 0)]

  if (cost && cost.tokens !== null && cost.tokens > 0) {
    parts.push(m.costTokens(compactNumber(cost.tokens)))
  }

  if (cost && cost.uses !== null) {
    parts.push(m.usage30d(compactNumber(cost.uses)))
  }

  return parts.join(', ')
}

export function statusLine(
  m: Translations['settings']['mcp'],
  status: ServerStatus,
  probe: Probe | undefined,
  server?: Record<string, unknown>,
  cost?: ServerCost
): string {
  switch (status) {
    case 'ok':
      return capabilitySummary(m, probe as McpTestResult, server, cost)

    case 'probing':
      return m.statusConnecting

    case 'needs-auth':
      return m.statusNeedsAuth

    case 'error':
      return m.statusError

    case 'off':
      return m.statusOff

    default:
      return ''
  }
}

/** The per-server cost overlay from a probe and the 30-day analytics map. */
export function serverCost(
  server: Record<string, unknown>,
  probe: Probe | undefined,
  name: string,
  toolCalls30d: null | Record<string, number>
): ServerCost {
  return {
    tokens: probe && probe !== 'probing' && probe.ok ? estimateServerTokens(server, probe.tools) : null,
    uses: toolCalls30d ? serverUsageCount(name, toolCalls30d) : null
  }
}
