import type { TranslationOverrides } from './define-locale'

export const frToolsets: NonNullable<TranslationOverrides['settings']>['toolsets'] = {
  loadingConfig: 'Chargement de la configuration',
  savedTitle: 'Identifiant enregistré',
  savedMessage: key => `${key} mis à jour.`,
  removedTitle: 'Identifiant supprimé',
  removedMessage: key => `${key} supprimé.`,
  failedSave: key => `Échec de l'enregistrement de ${key}`,
  failedRemove: key => `Échec de la suppression de ${key}`,
  failedReveal: key => `Échec de l'affichage de ${key}`,
  removeConfirm: key => `Supprimer ${key} de .env ?`,
  set: 'Définir',
  notSet: 'Non défini',
  selectedTitle: 'Fournisseur sélectionné',
  selectedMessage: provider => `${provider} est maintenant actif.`,
  failedSelect: provider => `Échec de la sélection de ${provider}`,
  failedLoad: 'Échec du chargement de la configuration des outils',
  noProviderOptions:
    "Cet ensemble d'outils n'a pas d'options de fournisseur — activez-le et il fonctionne avec votre configuration actuelle.",
  noProviders: "Aucun fournisseur n'est disponible pour cet ensemble d'outils pour le moment.",
  ready: 'Prêt',
  needsSignIn: 'Nécessite une connexion',
  needsSetup: 'Nécessite une configuration',
  activeBackend: 'Actif',
  browserAxesHint:
    'Browser Use sélectionne le pilote du navigateur. Le fournisseur détermine où le navigateur s’exécute. Les deux peuvent être actifs à la fois.',
  activeBackendHint: "Il s'agit de votre backend actif",
  useBackend: 'Utiliser ce backend',
  nousIncluded: 'Inclus avec un abonnement Nous — connectez-vous au portail Nous pour activer.',
  nousAuthNeededTitle: 'Se connecter au portail Nous',
  nousAuthNeededMessage: provider =>
    `${provider} est enregistré mais ne s'activera pas tant que vous ne vous serez pas connecté au portail Nous.`,
  nousAuthSignIn: 'Se connecter',
  nousAuthDoneTitle: 'Portail Nous connecté',
  nousAuthDoneMessage: "Vos backends d'abonnement sont maintenant actifs.",
  nousAuthFailed: "La connexion au portail Nous n'a pas été terminée",
  nousAuthFailedMessage: 'Réessayez.',
  nousAuthTryAgain: 'Réessayer',
  noApiKeyRequired: 'Aucune clé API requise.',
  postSetupHint: step =>
    `Ce backend nécessite une installation unique (${step}). S'exécute sur cette machine — peut prendre quelques minutes.`,
  postSetupInstalledHint: 'Installé. Relancez la configuration uniquement si quelque chose est cassé.',
  postSetupRun: 'Exécuter la configuration',
  postSetupRerun: 'Relancer la configuration',
  postSetupInstalled: 'Installé',
  postSetupRunning: 'Installation…',
  postSetupStarting: 'Démarrage…',
  postSetupCompleteTitle: 'Configuration terminée',
  postSetupCompleteMessage: step => `${step} installé.`,
  postSetupErrorTitle: 'Configuration terminée avec des erreurs',
  postSetupErrorMessage: step => `Consultez le journal ${step}.`,
  postSetupOpenLogs: 'Ouvrir les journaux',
  postSetupRunAgain: 'Relancer',
  postSetupFailed: step => `Échec de l'exécution de la configuration ${step}`,
  webSearchActive: backend => `Recherche : ${backend}`,
  webExtractActive: backend => `Extraction : ${backend}`,
  webCapabilityUnset: 'non défini',
  webUseForSearch: 'Utiliser pour la recherche',
  webUseForExtract: "Utiliser pour l'extraction",
  webUsedForSearch: 'Backend de recherche',
  webUsedForExtract: "Backend d'extraction",
  webCapabilitySelectedMessage: (provider, capability) =>
    `${provider} gère maintenant ${capability === 'search' ? 'la recherche web' : 'l’extraction de contenu web'}.`,
  failedSelectCapability: provider => `Échec de la définition de ${provider}`,
  loadingModels: 'Chargement du catalogue de modèles...',
  modelSectionTitle: 'Modèle',
  modelCount: count => `${count} modèle${count === 1 ? '' : 's'}`,
  modelInUse: 'En utilisation',
  modelDefault: 'par défaut',
  modelInactiveHint: "Sélectionnez d'abord ce backend pour changer son modèle.",
  modelSelectedTitle: 'Modèle sélectionné',
  modelSelectedMessage: model => `${model} s'applique aux nouvelles sessions.`,
  failedSelectModel: model => `Échec de la sélection de ${model}`,
  terminalBackend: {
    sectionTitle: "Backend d'exécution",
    loading: "Vérification des backends d'exécution…",
    failedLoad: 'Impossible de charger les backends de terminal',
    ready: 'Prêt',
    needsSetup: 'Nécessite une configuration',
    unavailable: 'Indisponible',
    inUse: 'En utilisation',
    selectedTitle: 'Backend sélectionné',
    selectedMessage: backend =>
      `Les commandes de terminal s'exécutent maintenant via ${backend}. S'applique aux nouvelles sessions.`,
    failedSelect: backend => `Échec de la sélection de ${backend}`,
    needsSetupHint:
      'Ce backend est sélectionné sans configuration complète — les commandes échoueront tant que la configuration n’est pas terminée.',
    needsSetupConfirmTitle: (backend: string) => `Sélectionner ${backend} quand même ?`,
    needsSetupConfirmDescription: (detail: string) =>
      `${detail} Les sessions démarrées après ce changement n’auront ni outils de terminal ni outils de fichiers tant que la configuration n’est pas terminée.`,
    needsSetupConfirmDescriptionGeneric:
      'Ce backend n’est pas encore configuré. Les sessions démarrées après ce changement n’auront ni outils de terminal ni outils de fichiers tant que la configuration n’est pas terminée.',
    needsSetupConfirmAction: 'Sélectionner quand même',
    unavailableTitle: 'Commandes de terminal indisponibles',
    unavailableMessage: backend =>
      `Hermes ne peut pas exécuter de commandes shell pour le moment : ${backend} n'est pas prêt. Passez en Local ou terminez la configuration de ${backend}, puis réessayez.`,
    openBackendSettings: 'Ouvrir les paramètres du terminal',
    useLocal: 'Utiliser Local',
    switchedToLocal:
      "Les commandes de terminal s'exécutent maintenant localement. S'applique aux nouvelles sessions."
  },
  browserRealProfile: {
    label: 'Utiliser mon profil de navigateur réel',
    description:
      "Copie les identifiants de connexion et les cookies de votre navigateur par défaut dans un instantané géré que l'agent utilise pour naviguer. Votre profil actif n'est jamais ouvert directement. S'applique aux nouvelles sessions.",
    enabledTitle: 'Navigation avec le profil réel activée',
    enabledMessage: 'Les nouvelles sessions utiliseront un instantané de votre profil de navigateur par défaut.',
    disabledTitle: 'Navigation avec le profil réel désactivée',
    disabledMessage:
      "L'instantané du profil sera supprimé ; les nouvelles sessions utiliseront un navigateur vierge.",
    failedSave: "Impossible d'enregistrer le paramètre du profil réel",
    prompt: {
      title: 'Restez connecté à vos sites',
      body: "Autorisez Hermes à naviguer avec un instantané de votre profil de navigateur par défaut afin que les sites s'ouvrent avec vos sessions déjà connectées.",
      bulletSnapshot: 'Les cookies et identifiants de connexion sont copiés dans un instantané géré.',
      bulletLiveProfile: "Votre profil de navigateur actif n'est jamais ouvert directement.",
      bulletLocal: 'Rien ne quitte cet ordinateur.',
      dontShowAgain: 'Ne plus afficher',
      notNow: 'Pas maintenant',
      enable: 'Utiliser mon profil'
    }
  }
}
