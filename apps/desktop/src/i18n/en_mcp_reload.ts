import type { Translations } from './types'

export const enMcpReload: Translations['connectorsPage']['reload'] = {
  action: 'Reload MCP servers',
  title: 'Reload MCP servers?',
  description:
    "This reconciles this profile's MCP servers with config and refreshes this chat's tools. The next message re-sends the full prompt, which costs prompt-cache tokens.",
  confirm: 'Reload now',
  deferred:
    'MCP servers updated for this profile. This chat keeps its tools until its running turn finishes; run the reload again then (or reopen the chat).',
  unchanged: 'MCP servers already up to date',
  success: 'MCP servers reloaded',
  failed: servers => `MCP servers reloaded, but these did not connect: ${servers}`,
  error: 'MCP servers were not reloaded',
  confirmRequired: '/reload-mcp requires confirmation',
  reloadedAlways: 'MCP servers reloaded · /reload-mcp will no longer ask for confirmation in this profile',
  notConnected: servers => `Did not connect: ${servers}`,
  completionNow: 'Reload now (skip the confirmation once)',
  completionAlways: 'Reload now and stop asking in this profile',
  skillsRemote:
    'Skills list refreshed. The live rescan runs on the gateway that owns this profile, not from this window.',
  skillsFailed: 'Skills failed to reload'
}
