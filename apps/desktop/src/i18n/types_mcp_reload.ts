/** Connectors ▸ Reload MCP servers, plus the `/reload-mcp` and `/reload-skills` output it shares. */
export interface McpReloadCopy {
  action: string
  title: string
  description: string
  confirm: string
  success: string
  failed: (servers: string) => string
  error: string
  confirmRequired: string
  reloadedAlways: string
  notConnected: (servers: string) => string
  completionNow: string
  completionAlways: string
  skillsFailed: string
}
