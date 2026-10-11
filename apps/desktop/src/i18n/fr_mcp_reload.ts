import type { TranslationOverrides } from './define-locale'

export const frMcpReload: NonNullable<TranslationOverrides['connectorsPage']>['reload'] = {
  action: 'Recharger les serveurs MCP',
  title: 'Recharger les serveurs MCP ?',
  description:
    "Cela redémarre les serveurs MCP de ce profil et le prochain message renvoie l'invite complète, ce qui consomme des jetons du cache d'invite.",
  confirm: 'Recharger maintenant',
  success: 'Serveurs MCP rechargés',
  failed: servers => `Serveurs MCP rechargés, mais ceux-ci ne se sont pas connectés : ${servers}`,
  error: "Les serveurs MCP n'ont pas été rechargés",
  confirmRequired: '/reload-mcp nécessite une confirmation',
  reloadedAlways: 'Serveurs MCP rechargés · /reload-mcp ne demandera plus de confirmation dans ce profil',
  notConnected: servers => `Non connectés : ${servers}`,
  completionNow: 'Recharger maintenant (ignorer la confirmation une fois)',
  completionAlways: 'Recharger maintenant et ne plus demander dans ce profil',
  skillsFailed: 'Échec du rechargement des skills'
}
