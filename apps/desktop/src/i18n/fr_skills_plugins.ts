import type { TranslationOverrides } from './define-locale'

/** French copy for the Skills ▸ Plugins tab; `skills.plugins`, composed by fr.ts. */
export const frSkillsPlugins: NonNullable<NonNullable<TranslationOverrides['skills']>['plugins']> = {
  agentTitle: "Plugins de l'agent",
  agentBlurb:
    "Étendez l'agent du profil sélectionné avec des outils, hooks et fournisseurs. Les changements prennent effet après le redémarrage du gateway.",
  pageBlurb: 'Un plugin peut étendre cette application, l’agent ou les deux — chaque moitié a son propre interrupteur.',
  halfDesktop: 'Desktop',
  halfDesktopHint: 'cette application, identique pour tous les profils',
  halfAgent: 'Agent',
  halfAgentIn: profile => `Agent dans ${profile}`,
  defaultProfile: 'Hermes (par défaut)',
  kindAgent: 'Agent',
  kindDesktop: 'Desktop',
  kindBoth: 'Agent + Desktop',
  installAgentHere: 'Installer ici',
  installAgentHereTip: profile =>
    `La partie Desktop est chargée dans cette application, mais la partie agent n'est pas installée dans ${profile}. Installez-la dans ce profil.`,
  installAgentHereNoOrigin:
    "La partie agent n'est pas installée dans ce profil et ce paquet a été copié manuellement, sans entrée de catalogue ni dépôt Git. Copiez son dossier dans le profil ou réinstallez-le depuis Git.",
  desktopHalfPending: 'copie…',
  desktopHalfPendingTip:
    "Ce paquet contient une partie Desktop qui n'a pas encore été copiée dans l'application. Relancez l'analyse ou redémarrez l'application.",
  desktopHalfRemote: 'indisponible (backend distant)',
  desktopHalfRemoteTip:
    'La moitié bureau de ce paquet se trouve sur le disque du backend distant, que cette application ne peut pas lire. Pour l’utiliser ici, lancez Installer depuis Git avec l’URL du dépôt du paquet et la cible Bureau cochée — cela clone la moitié bureau sur cette machine.',
  emptyAll: 'Aucun plugin pour le moment.',
  empty: "Aucun plugin d'agent installé pour ce profil.",
  emptyHint: 'Parcourez le catalogue ci-dessous pour installer un plugin vérifié en un clic.',
  loadFailed: "Impossible de charger les plugins de l'agent",
  toggleFailed: name => `Impossible de modifier l'état de ${name}`,
  toolsetOn: (name: string, profile: string) => `Outils agent de ${name} activés pour ${profile}`,
  toolsetOff: (name: string, profile: string) => `Outils agent de ${name} désactivés pour ${profile}`,
  toolsetToggleFailed: (name: string) =>
    `Impossible de modifier les outils agent de ${name} ; le panneau Desktop reste inchangé`,
  legacyBackend: 'Ce backend est trop ancien pour gérer les plugins depuis cet écran ; mettez Hermes à jour.',
  portableBadge: 'portable',
  serverStates: {
    connected: 'connecté',
    app_not_running: 'application non lancée',
    hermes_not_connected: 'connexion MCP manquante',
    endpoint_unavailable: 'point de terminaison indisponible',
    no_interactive_session: 'aucune session interactive',
    version_too_old: 'version trop ancienne',
    missing_app: 'application manquante',
    unsupported_gpu: 'GPU non prise en charge',
    unknown: 'état inconnu'
  },
  catalogTitle: 'Catalogue de plugins',
  catalogBrowse: 'Parcourir',
  catalogHide: 'Masquer le catalogue',
  catalogHint:
    'Utilisez « + Ajouter à cet agent » sur un plugin : les entrées vérifiées sont installées depuis leur commit épinglé dans le profil sélectionné.',
  alreadyInstalled: (name: string) => `${name} est déjà installé dans ce profil.`,
  catalogProvenance: (sha: string) => `Installé depuis le catalogue Hermes${sha ? ` au commit épinglé ${sha}` : ''}.`,
  pinnedProvenance: (sha: string) =>
    `Épinglé au commit ${sha}. Les mises à jour sont refusées tant qu’il n’est pas réinstallé avec un nouvel épinglage.`,
  pinnedBadge: (sha: string) => `épinglé @ ${sha}`,
  update: 'Mettre à jour',
  updateToPin: (sha: string) => `Mettre à jour vers ${sha}`,
  updateFailed: (name: string) => `Impossible de mettre à jour ${name}`,
  updated: (name: string) => `${name} mis à jour. Redémarrez le gateway pour l’appliquer.`,
  upToDate: (name: string) => `${name} est déjà à jour.`,
  updateConsentTitle: (name: string) => `${name} attend votre accord`,
  updateConsentBody: (name: string) =>
    `La mise à jour de ${name} apporte des changements à examiner. Appliquez-la uniquement si vous faites confiance à la source :`,
  updateConsentConfirm: 'Appliquer la mise à jour',
  uninstall: 'Désinstaller',
  uninstallTip: (name: string, profile: string) => `Désinstaller ${name} de ${profile}`,
  uninstallConfirmTitle: (name: string) => `Désinstaller ${name} ?`,
  uninstallConfirmBody: (name: string, profile: string) =>
    `Cette action supprime les fichiers du plugin du profil ${profile}. Toute moitié bureau fournie est supprimée avec lui. Réinstallez-le depuis le catalogue ou depuis Git à tout moment.`,
  uninstallFailed: (name: string) => `Impossible de désinstaller ${name}`,
  uninstalled: (name: string) => `${name} désinstallé. Redémarrez le gateway pour le décharger.`,
  uninstallDesktopTip: (name: string) => `Désinstaller ${name} de cette application`,
  uninstallDesktopConfirmBody: (name: string) =>
    `Cette action supprime ${name} du dossier desktop-plugins de cet ordinateur et le décharge immédiatement. Réinstallez-le depuis Git ou remettez le dossier en place à tout moment.`,
  uninstalledDesktop: (name: string) => `${name} désinstallé.`,
  deepLinkErrorTitle: 'Lien d’installation de plugin refusé',
  deepLinkCatalogInvalidName: 'Le nom de catalogue du lien est manquant ou invalide.',
  deepLinkCatalogUnknown: (name: string) =>
    `\u00AB\u00A0${name}\u00A0\u00BB ne figure pas dans le catalogue de plugins Hermes. Rien n’a été installé.`,
  deepLinkCatalogUnavailable:
    'Impossible de charger le catalogue de plugins Hermes. Vérifiez votre connexion et rouvrez le lien.',
  settingsToggle: (name: string) => `Paramètres : ${name}`,
  settingsForm: {
    save: 'Enregistrer les paramètres',
    saved: (name: string) => `Paramètres de ${name} enregistrés.`,
    saveFailed: (name: string) => `Impossible d’enregistrer les paramètres de ${name}`,
    required: 'Obligatoire',
    secretSet: '•••••••• (défini)',
    secretStoredAs: (env: string) =>
      `Stocké dans le .env du profil sous ${env}, jamais dans config.yaml ; laissez vide pour conserver la valeur actuelle.`
  }
}
