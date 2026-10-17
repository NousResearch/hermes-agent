import type { TranslationOverrides } from './define-locale'
import { frConnectorAccounts } from './fr_connector_accounts'

export const frMcpReload: NonNullable<TranslationOverrides['connectorsPage']>['reload'] = {
  action: 'Recharger les serveurs MCP',
  title: 'Recharger les serveurs MCP ?',
  description:
    "Cela réconcilie les serveurs MCP de ce profil avec la configuration. Si les outils de ce chat changent, le prochain message démarre un nouveau préfixe de cache d'invite ; un rechargement sans changement ne renvoie pas l'invite.",
  confirm: 'Recharger maintenant',
  deferred:
    "Serveurs MCP mis à jour pour ce profil. Ce chat conserve ses outils jusqu'à la fin de son tour en cours ; relancez le rechargement ensuite (ou rouvrez le chat).",
  unchanged: 'Les serveurs MCP sont déjà à jour',
  success: 'Serveurs MCP rechargés',
  failed: servers => `Serveurs MCP rechargés, mais ceux-ci ne se sont pas connectés : ${servers}`,
  error: "Les serveurs MCP n'ont pas été rechargés",
  confirmRequired: '/reload-mcp nécessite une confirmation',
  reloadedAlways: 'Serveurs MCP rechargés · /reload-mcp ne demandera plus de confirmation dans ce profil',
  notConnected: servers => `Non connectés : ${servers}`,
  completionNow: 'Recharger maintenant (ignorer la confirmation une fois)',
  completionAlways: 'Recharger maintenant et ne plus demander dans ce profil',
  skillsRemote:
    "Liste des skills actualisée. La nouvelle analyse en direct s'exécute sur la passerelle propriétaire de ce profil, pas depuis cette fenêtre.",
  skillsFailed: 'Échec du rechargement des skills'
}

/** The connectors-page slice this file owns: the accounts list plus the reload copy, so the
 *  locale facade stays one spread (its line budget only goes down). */
export const frConnectorPage: NonNullable<TranslationOverrides['connectorsPage']> = {
  accounts: frConnectorAccounts,
  reload: frMcpReload
}
