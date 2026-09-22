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

  // An explicit `include` (even []) is a whitelist — the runtime registers nothing for `[]`
  // (tools/mcp_tool_registration.py), so the desktop must not show every tool as enabled (#12865).
  if (include !== undefined) {
    return include.includes(name)
  }

  return !exclude?.includes(name)
}

// Toggle one tool, preserving the config's mode (include if the key is present, even empty, else
// an exclude denylist). An emptied exclude is dropped; an emptied include is kept (block-all).
export function toggleToolInServer(server: ServerConfig, name: string): ServerConfig {
  const { exclude, include } = readToolsFilter(server)
  const key = include !== undefined ? 'include' : 'exclude'
  const current = (key === 'include' ? include : exclude) ?? []
  const names = current.includes(name) ? current.filter(n => n !== name) : [...current, name]
  const tools = { ...toolsObject(server) }

  if (key === 'include') {
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
