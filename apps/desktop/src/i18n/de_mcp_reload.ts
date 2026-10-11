import type { TranslationOverrides } from './define-locale'

export const deMcpReload: NonNullable<TranslationOverrides['connectorsPage']>['reload'] = {
  action: 'MCP-Server neu laden',
  title: 'MCP-Server neu laden?',
  description:
    'Dadurch werden die MCP-Server dieses Profils neu gestartet und die nächste Nachricht sendet den vollständigen Prompt erneut — das kostet Prompt-Cache-Token.',
  confirm: 'Jetzt neu laden',
  success: 'MCP-Server neu geladen',
  failed: servers => `MCP-Server neu geladen, aber diese konnten keine Verbindung herstellen: ${servers}`,
  error: 'MCP-Server wurden nicht neu geladen',
  confirmRequired: '/reload-mcp erfordert eine Bestätigung',
  reloadedAlways: 'MCP-Server neu geladen · /reload-mcp fragt in diesem Profil nicht mehr nach Bestätigung',
  notConnected: servers => `Keine Verbindung: ${servers}`,
  completionNow: 'Jetzt neu laden (Bestätigung einmal überspringen)',
  completionAlways: 'Jetzt neu laden und in diesem Profil nicht mehr fragen',
  skillsFailed: 'Skills konnten nicht neu geladen werden'
}
