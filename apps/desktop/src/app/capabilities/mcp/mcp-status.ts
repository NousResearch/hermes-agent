
import { compactNumber } from '@hermes/shared'

import { getUsageAnalytics, type McpTestResult, type ProfileScope } from '@/hermes'
import type { Translations } from '@/i18n'
import { estimateServerTokens, serverUsageCount } from '@/lib/mcp-cost'
import { NEEDS_AUTH_RE } from '@/lib/mcp-probe-cache'
import type { McpServerEntry } from '@/lib/mcp-servers'
import { countEnabledTools } from '@/lib/mcp-tool-filter'

import { serverEnabled } from './mcp-doc'

export const MCP_CATALOG_KEY = ['mcp-catalog'] as const

export type Probe = McpTestResult | 'probing'

export type ServerStatus = 'error' | 'needs-auth' | 'off' | 'ok' | 'probing' | 'unknown'

export const okProbe = (probe: Probe | undefined): McpTestResult | null =>
  probe && probe !== 'probing' && probe.ok ? probe : null

export interface ServerCost {
  tokens: null | number
  uses: null | number
}

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
    return null
  }
}

export function statusOf(server: McpServerEntry, probe: Probe | undefined): ServerStatus {
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

export const STATUS_DOT = {
  ok: 'bg-emerald-500',
  error: 'bg-red-500',
  'needs-auth': 'bg-amber-500',
  probing: 'animate-pulse bg-foreground/40',
  off: 'bg-foreground/20',
  unknown: 'bg-foreground/20'
} satisfies Record<ServerStatus, string>

export function canAuthenticate(server: McpServerEntry, status: ServerStatus): boolean {
  const hasHeaderAuth = server.headers instanceof Object

  return (
    // oxlint-disable-next-line anti-slop/no-runtime-typeof -- SAFETY: `server` is a hand-editable mcp.json entry; this is where its `url` becomes a string.
    typeof server.url === 'string' &&
    !hasHeaderAuth &&
    (server.auth === 'oauth' ? status === 'needs-auth' || status === 'error' : !server.auth && status === 'needs-auth')
  )
}

export function capabilitySummary(
  m: Translations['settings']['mcp'],
  probe: McpTestResult,
  server?: McpServerEntry,
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
  server?: McpServerEntry,
  cost?: ServerCost
): string {
  switch (status) {
    case 'ok': {
      const ok = okProbe(probe)

      return ok ? capabilitySummary(m, ok, server, cost) : ''
    }

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

export function serverCost(
  server: McpServerEntry,
  probe: Probe | undefined,
  name: string,
  toolCalls30d: null | Record<string, number>
): ServerCost {
  const ok = okProbe(probe)

  return {
    tokens: ok ? estimateServerTokens(server, ok.tools) : null,
    uses: toolCalls30d ? serverUsageCount(name, toolCalls30d) : null
  }
}
