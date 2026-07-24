// Per-tool MCP gating. A server's optional `tools.include` (whitelist) /
// `tools.exclude` (denylist) decide which discovered tools the agent registers
// — `include` wins, no filter means all. Mirrors `_register_server_tools` in
// `tools/mcp_tool.py`.

export interface McpToolsFilter {
  exclude?: string[]
  include?: string[]
}

type ServerConfig = Record<string, unknown>

const asNames = (value: unknown): string[] | undefined =>
  Array.isArray(value) ? value.filter((v): v is string => typeof v === 'string') : undefined

const toolsObject = (server: ServerConfig | null | undefined): Record<string, unknown> => {
  const tools = server?.tools

  return tools && typeof tools === 'object' && !Array.isArray(tools) ? (tools as Record<string, unknown>) : {}
}

export function readToolsFilter(server: ServerConfig | null | undefined): McpToolsFilter {
  const tools = toolsObject(server)

  return { exclude: asNames(tools.exclude), include: asNames(tools.include) }
}

export function isToolEnabled(server: ServerConfig | null | undefined, name: string): boolean {
  const { exclude, include } = readToolsFilter(server)

  // Explicit `include` (including []) is a whitelist: empty blocks all tools.
  // Missing/undefined include falls back to exclude denylist or "all on".
  // Mirrors tools/mcp_tool.py `_normalize_name_filter` / `_should_register`.
  if (include !== undefined) {
    return include.includes(name)
  }

  return !exclude?.includes(name)
}

// Toggle one tool, preserving the config's mode (include if key was present,
// even when empty, else an exclude denylist).
// Empty exclude is dropped; empty include is retained (`include: []` = block all).
export function toggleToolInServer(server: ServerConfig, name: string): ServerConfig {
  const { exclude, include } = readToolsFilter(server)
  const key = include !== undefined ? 'include' : 'exclude'
  const current = (key === 'include' ? include : exclude) ?? []
  const names = current.includes(name) ? current.filter(n => n !== name) : [...current, name]
  const tools = { ...toolsObject(server) }

  if (key === 'include') {
    // Keep empty include so reconfiguration / registration treat it as block-all.
    tools.include = names
  } else if (names.length) {
    tools.exclude = names
  } else {
    delete tools.exclude
  }

  const next = { ...server }

  if (Object.keys(tools).length) {
    next.tools = tools
  } else {
    delete next.tools
  }

  return next
}

export const countEnabledTools = (server: ServerConfig | null | undefined, names: string[]): number =>
  names.filter(name => isToolEnabled(server, name)).length
