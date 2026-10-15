import type { CatalogInstallTranslations } from './types_catalog_install'

export const frCatalogInstall: CatalogInstallTranslations = {
  preparing: 'Préparation de l’installation…',
  install: 'Installer',
  advanced: 'Avancé',
  skip: 'Ignorer',
  installing: 'Installation…',
  installed: 'Installé',
  notInstalled: 'Non installé',
  failed: 'Échec',
  showNames: 'afficher les noms',
  hideNames: 'masquer les noms',
  skill: name => `skill ${name}`,
  kind: { plugin: 'plugin', skill: 'skill' },
  tier: { official: 'officiel', community: 'communauté' },
  targetProfile: profile => `S’installe dans votre profil ${profile}`,
  sendFailed: 'Impossible d’envoyer votre réponse. Réessayez.',
  commitLabel: 'Commit',
  subdirLabel: 'Dossier',
  securityHeading: 'Sécurité',
  scan: { passed: 'Analyse réussie', warnings: 'L’analyse a relevé des avertissements', failed: 'Échec de l’analyse' },
  requirementsLabel: 'Nécessite',
  requiresHermes: range => `Hermes ${range}`,
  envVar: name => `variable d’environnement ${name}`,
  credentialsHeading: 'Identifiants',
  phase: {
    downloading: 'Téléchargement en cours…',
    python_packages: 'Installation des paquets Python…',
    loading_tools: 'Chargement de ses outils…'
  },
  serverNotConnected: (server, reason) => `Serveur MCP ${server} non connecté${reason ? `: ${reason}` : ''}`,
  notEnabled: 'Installé mais pas activé',
  missingEnv: names => `Définissez ${names} pour terminer la configuration`,
  alreadyInstalled: 'Déjà installé ; laissé tel quel'
}
