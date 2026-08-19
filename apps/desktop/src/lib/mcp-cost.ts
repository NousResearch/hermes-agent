// MCP fleet cost/usage math — pure functions behind the Capabilities page's
// per-server cost overlay ("~4.2k tok, 3 uses/30d"). Display-only: nothing
// here changes what schemas are sent to models.

import { isToolEnabled } from '@/lib/mcp-tool-filter'

/** One probed tool as the desktop's /test endpoint reports it. `schema_chars`
 *  (the JSON-stringified registry schema's length) is additive-optional —
 *  older backends omit it and the UI simply shows no token estimate. */
export interface McpProbedTool {
  description?: string
  name: string
  schema_chars?: number
}

/**
 * Approximate per-call prompt-token cost of a server's tool schemas: every
 * enabled tool's schema rides along on every model call, so the sum of
 * `ceil(schema_chars / 4)` over ENABLED tools (per the server's
 * `tools.include`/`tools.exclude` filter) is what the server "adds per call".
 *
 * Returns `null` when no enabled tool carries `schema_chars` (older backend,
 * failed probe, or everything filtered out) — the caller shows no estimate.
 */
export function estimateServerTokens(
  server: Record<string, unknown> | null | undefined,
  tools: McpProbedTool[]
): null | number {
  let total = 0
  let sawSchema = false

  for (const tool of tools) {
    if (typeof tool.schema_chars !== 'number' || !Number.isFinite(tool.schema_chars) || tool.schema_chars <= 0) {
      continue
    }

    if (!isToolEnabled(server, tool.name)) {
      continue
    }

    sawSchema = true
    total += Math.ceil(tool.schema_chars / 4)
  }

  return sawSchema ? total : null
}

/** Mirror of `sanitize_mcp_name_component` in tools/mcp_tool.py: hyphens (and
 *  anything else outside `[A-Za-z0-9_]`) become underscores. */
export const sanitizeMcpNameComponent = (value: string): string => value.replace(/[^A-Za-z0-9_]/g, '_')

/** Registry-name prefix for one server's tools. MCP tools register (and show
 *  up in usage analytics) as `mcp__<server>__<tool>` — the convention pinned
 *  by `mcp_prefixed_tool_name` in tools/mcp_tool.py. */
export const mcpServerUsagePrefix = (serverName: string): string => `mcp__${sanitizeMcpNameComponent(serverName)}__`

/**
 * Sum 30-day analytics call counts across one server's tools (including the
 * per-server utility tools like `…__list_resources`, which also carry the
 * server prefix). Analytics keys are the registry names, so a prefix match is
 * the same grouping the agent's own registry uses.
 */
export function serverUsageCount(serverName: string, toolCalls: Record<string, number>): number {
  const prefix = mcpServerUsagePrefix(serverName)
  let total = 0

  for (const [tool, count] of Object.entries(toolCalls)) {
    if (tool.startsWith(prefix) && typeof count === 'number' && Number.isFinite(count)) {
      total += count
    }
  }

  return total
}
