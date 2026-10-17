import { deConnectorAccounts } from './de_connector_accounts'
import type { TranslationOverrides } from './define-locale'

export const deMcpReload: NonNullable<TranslationOverrides['connectorsPage']>['reload'] = {
  action: 'MCP-Server neu laden',
  title: 'MCP-Server neu laden?',
  description:
    'Dadurch werden die MCP-Server dieses Profils mit der Konfiguration abgeglichen und die Tools dieses Chats aktualisiert. Die nächste Nachricht sendet den vollständigen Prompt erneut — das kostet Prompt-Cache-Token.',
  confirm: 'Jetzt neu laden',
  deferred:
    'MCP-Server für dieses Profil aktualisiert. Dieser Chat behält seine Tools, bis sein laufender Turn endet; lade danach erneut (oder öffne den Chat neu).',
  unchanged: 'MCP-Server sind bereits aktuell',
  success: 'MCP-Server neu geladen',
  failed: servers => `MCP-Server neu geladen, aber diese konnten keine Verbindung herstellen: ${servers}`,
  error: 'MCP-Server wurden nicht neu geladen',
  confirmRequired: '/reload-mcp erfordert eine Bestätigung',
  reloadedAlways: 'MCP-Server neu geladen · /reload-mcp fragt in diesem Profil nicht mehr nach Bestätigung',
  notConnected: servers => `Keine Verbindung: ${servers}`,
  completionNow: 'Jetzt neu laden (Bestätigung einmal überspringen)',
  completionAlways: 'Jetzt neu laden und in diesem Profil nicht mehr fragen',
  skillsRemote:
    'Skill-Liste aktualisiert. Der Live-Rescan läuft auf dem Gateway, dem dieses Profil gehört, nicht aus diesem Fenster.',
  skillsFailed: 'Skills konnten nicht neu geladen werden'
}

/** The connectors-page slice this file owns: the accounts list plus the reload copy, so the
 *  locale facade stays one spread (its line budget only goes down). */
export const deConnectorPage: NonNullable<TranslationOverrides['connectorsPage']> = {
  accounts: deConnectorAccounts,
  reload: deMcpReload
}
