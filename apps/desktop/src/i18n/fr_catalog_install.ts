import type { CatalogInstallTranslations } from './types_catalog_install'

export const frCatalogInstall: CatalogInstallTranslations = {
  preparing: 'Préparation de l\'installation…',
  install: 'Installer',
  advanced: 'Avancé',
  skip: 'Ignorer',
  installing: 'Installation en cours…',
  installed: 'Installé',
  notInstalled: 'Non installé',
  failed: 'Échoué',
  showNames: 'afficher les noms',
  hideNames: 'cacher les noms',
  skill: name => `compétence ${name}`,
  kind: { plugin: 'extension', skill: 'compétence' },
  tier: { official: 'officiel', community: 'communauté' },
  targetProfile: profile => `Installé dans votre profil ${profile}`,
  sendFailed: 'Impossible d\'envoyer votre réponse. Réessayez.',
  commitLabel: 'Valider',
  subdirLabel: 'Dossier',
  securityHeading: 'Sécurité',
  scan: { passed: 'Analyse réussie', warnings: 'Analyse a trouvé des avertissements', failed: 'Analyse échouée' },
  requirementsLabel: 'Nécessite',
  requiresHermes: range => `Hermes ${range}`,
  envVar: name => `${name} variable d\'environnement`,
  credentialsHeading: 'Identifiants',
  phase: {
    downloading: 'Téléchargement en cours…',
    python_packages: 'Installation des paquets Python…',
    loading_tools: 'Chargement des outils…'
  },
  serverNotConnected: (server, reason) => `Serveur MCP ${server} non connecté${reason ? `: ${reason}` : ''}`,
  notEnabled: 'Installé mais pas activé',
  missingEnv: names => `Définissez ${names} pour terminer la configuration`,
  alreadyInstalled: 'Déjà installé; laissé tel quel'
}