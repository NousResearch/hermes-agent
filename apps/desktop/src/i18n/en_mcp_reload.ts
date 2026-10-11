import type { Translations } from './types'

export const enMcpReload: Translations['connectorsPage']['reload'] = {
  action: 'Reload MCP servers',
  title: 'Reload MCP servers?',
  description:
    "This restarts this profile's MCP servers and the next message re-sends the full prompt, which costs prompt-cache tokens.",
  confirm: 'Reload now',
  success: 'MCP servers reloaded',
  failed: servers => `MCP servers reloaded, but these did not connect: ${servers}`,
  error: 'MCP servers were not reloaded',
  confirmRequired: '/reload-mcp requires confirmation',
  reloadedAlways: 'MCP servers reloaded · /reload-mcp will no longer ask for confirmation in this profile',
  notConnected: servers => `Did not connect: ${servers}`,
  completionNow: 'Reload now (skip the confirmation once)',
  completionAlways: 'Reload now and stop asking in this profile',
  skillsFailed: 'Skills failed to reload'
}
