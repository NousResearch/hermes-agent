import { defineFieldCopy } from '@/app/settings/field-copy'

import { defineLocale, type TranslationOverrides } from './define-locale'
import { introFr } from './intro-fr'

export const frOverrides = {
  intro: introFr,
  connectors: {
    title: 'Connectez vos applications',
    connect: 'Connecter',
    skip: 'Pas maintenant',
    cancel: "Arrêter l'attente",
    retry: 'Réessayer',
    grant: 'Reconnecter',
    connected: 'Connecté',
    checking: 'Vérification de vos applications…',
    notConnected: 'Non connecté',
    skipped: 'Ignoré',
    disabled: 'Indisponible',
    failed: 'Connexion impossible',
    needsAuth: 'Accès expiré',
    opening: 'Ouverture de la connexion…',
    waiting: 'Terminez la connexion dans votre navigateur…',
    timeout: "Toujours en attente de l'autorisation.",
    refresh: "Actualiser l'état",
    connectError: "Impossible de démarrer l'autorisation. Réessayez.",
    connectErrorFor: app => `Impossible de démarrer l'autorisation pour ${app}.`,
    unavailable: 'Les connecteurs ne sont pas disponibles pour cette session.',
    ownerMissing: 'Rouvrez cette conversation pour gérer ses connexions.',
    search: 'Rechercher une application',
    empty: 'Aucune application correspondante',
    disclaimer: "La connexion est facultative. N'autorisez que les applications que vous voulez confier à Hermes.",
    execution: 'Outils des connecteurs',
    setup: server => `Configurer ${server}`,
    openInBrowser: 'Ouvrir dans le navigateur',
    setupCancel: 'Annuler',
    authorizedToolsUnavailable: 'Autorisé. Outils indisponibles.',
    required: 'Obligatoire'
  },
  connectorsPage: {
    title: 'Connecteurs',
    searchPlaceholder: (count: number) => `Rechercher parmi ${count} applications`,
    filterCategory: 'Catégorie',
    categoryAll: 'Toutes les catégories',
    uncategorised: 'Sans catégorie',
    residencyLocal: 'Sur cet appareil',
    segment: {
      all: 'Tous',
      available: 'Disponibles',
      connected: 'Connectés',
      off: 'Désactivés'
    },
    group: {
      connected: 'Connectés',
      connectedNote: "Connexions en échec d'abord.",
      available: 'Disponibles',
      off: 'Désactivés',
      offNote: 'Les connexions sont conservées.'
    },
    card: {
      kindManaged: 'Géré',
      kindCatalog: 'MCP · Catalogue',
      kindCustom: 'MCP · Personnalisé',
      kindPlugin: (plugin: string) => `MCP · Plugin ${plugin}`,
      inCatalog: 'Dans le catalogue Hermes',
      hostedTwin: 'Version gérée disponible',
      alsoLocal: 'Fonctionne aussi sur cet appareil',
      open: (name: string) => `Ouvrir ${name}`,
      turnServerOn: (name: string) => `Activer ${name}`,
      turnServerOff: (name: string) => `Désactiver ${name}`,
      state: {
        accessExpired: 'Accès expiré',
        available: 'Disponible',
        connected: 'Connecté',
        connecting: 'Connexion',
        connectionUnknown: 'État inconnu',
        couldNotConnect: 'Connexion impossible',
        offByYourOrganisation: 'Désactivé par votre organisation',
        offForYou: 'Désactivé pour vous',
        serverConnecting: 'Connexion…',
        serverError: 'Erreur',
        serverNeedsAuth: 'Authentification requise',
        serverOff: 'Désactivé',
        serverOn: 'Activé',
        serverOnUnused: 'Activé, inutilisé'
      },
      fact: {
        tools: (count: number) => `${count} outil${count > 1 ? 's' : ''}`,
        toolsOff: (count: number) => `${count} outil${count > 1 ? 's' : ''} désactivé${count > 1 ? 's' : ''}`,
        toolsOn: (count: number) => `${count} outil${count > 1 ? 's' : ''} activé${count > 1 ? 's' : ''}`,
        toolsSomeOn: (total: number, on: number) => `${total} outils, ${on} activé${on > 1 ? 's' : ''}`
      },
      verb: {
        authenticate: "S'authentifier",
        connect: 'Connecter',
        install: 'Installer',
        openLogs: 'Ouvrir les journaux',
        reconnect: 'Reconnecter',
        stopWaiting: "Arrêter l'attente",
        tryAgain: 'Réessayer',
        turnBackOn: 'Réactiver'
      },
      reason: {
        finishSignIn: 'Terminez la connexion dans votre navigateur.',
        reconnect: 'Reconnectez-vous pour que cette application continue de fonctionner.',
        serverError: 'Le serveur a refusé la connexion.',
        serverNeedsAuth: 'Connectez-vous pour que ce serveur puisse répondre.'
      }
    },
    page: {
      loading: 'Lecture du catalogue et des serveurs de cet ordinateur',
      emptyTitle: 'Aucune application pour le moment. Ajoutez un serveur sur cet ordinateur pour commencer.',
      noMatchTitle: 'Aucune application correspondante',
      noMatchBody: 'Aucun résultat. Indiquez à Hermes votre propre serveur MCP pour l’ajouter.',
      clearSearch: 'Effacer la recherche',
      hostedFailedTitle: 'Impossible de joindre les applications hébergées.',
      hostedFailedBody:
        'Les serveurs de cet ordinateur ne sont pas affectés et fonctionnent toujours. Rien n’a été désactivé.',
      retry: 'Réessayer',
      matchesElsewhere: (count: number) =>
        `${count} autre${count > 1 ? 's' : ''} résultat${count > 1 ? 's' : ''} dans d’autres groupes.`,
      showAllMatches: 'Afficher tous les résultats',
      segmentNoMatch: (segment: string) => `Aucun résultat dans ${segment} : tous les résultats sont affichés.`,
      freeTierNote: 'Les connexions restent sur cet ordinateur jusqu’à ce que vous vous connectiez.',
      signInLine: 'Connectez-vous à Nous pour utiliser les applications gérées.',
      signIn: 'Se connecter',
      managedUnavailable: 'Les applications gérées ne sont pas encore disponibles pour ce compte.',
      writeFailed: 'Cette modification n’a pas été enregistrée.',
      refreshFailed: 'La liste des outils n’a pas été actualisée.',
      disconnectNoAccount: 'Hermes n’a aucun compte à déconnecter ici. Actualisez la page et réessayez.',
      disconnectRefused:
        'Nous ne peut pas supprimer cette connexion pour le moment. Désactivez plutôt l’application avec l’interrupteur, ou réessayez plus tard.'
    },
    add: {
      action: 'Ajouter le vôtre',
      title: 'Se connecter à un MCP personnalisé',
      hint: 'une nouvelle entrée dans mcp.json sur cet appareil',
      pasteLabel: 'Collez une commande ou un extrait',
      pastePlaceholder: 'npx -y @modelcontextprotocol/server-filesystem /chemin/vers/dossier',
      pasteNoMatch: 'Rien ici ne ressemble à un serveur. Remplissez plutôt les champs ci-dessous.',
      name: 'Nom',
      nameTaken: 'Ce nom est déjà utilisé.',
      type: 'Type',
      typeStdio: 'STDIO',
      typeHttp: 'HTTP streamable',
      command: 'Commande de lancement',
      args: 'Arguments',
      addArg: '+ Ajouter un argument',
      envVars: "Variables d'environnement",
      addEnvVar: "+ Ajouter une variable d'environnement",
      passthrough: "Transmission des variables d'environnement",
      addPassthrough: '+ Ajouter une variable',
      cwd: 'Répertoire de travail',
      url: 'URL',
      headers: 'En-têtes',
      addHeader: '+ Ajouter un en-tête',
      auth: 'Authentification',
      authNone: 'Aucune',
      authOauth: 'OAuth',
      authBearer: 'Jeton Bearer',
      keyPlaceholder: 'CLÉ',
      valuePlaceholder: 'valeur',
      removeRow: 'Supprimer cette ligne',
      editJson: 'Modifier mcp.json',
      saveFailed: "Ce serveur n'a pas été enregistré."
    },
    dialog: {
      disconnect: 'Déconnecter',
      disconnectTitle: (name: string) => `Déconnecter ${name} ?`,
      disconnectBody: "Hermes cesse d'agir avec ce compte. Vous pouvez vous reconnecter à tout moment.",
      menuRefreshTools: 'Actualiser les outils',
      moreActions: "Plus d'actions",
      removeServerTitle: (name: string) => `Supprimer ${name} ?`,
      removeServerBody: "L'entrée est retirée de mcp.json sur cet ordinateur. Rien d'autre n'est supprimé.",
      appSwitch: (name: string) => `Hermes peut utiliser ${name}`,
      waysTitle: (name: string) => `Où ${name} s'exécute`,
      wayNotConnected: (name: string) => `Pas encore connecté. Connectez-vous à ${name} dans votre navigateur.`,
      wayHosted: 'Géré',
      bothOn: (name: string) => `Les deux sont activés : Hermes voit donc chaque outil ${name} en double.`,
      turnOffLocal: 'Désactiver le serveur local',
      providedByPlugin: (plugin: string) => `Fourni par le plugin ${plugin}`,
      openPlugins: "Ouvrir l'onglet Plugins",
      nousLine: 'Les applications Nous suivent votre compte, pas le profil.',
      rulesReadOnly: 'Les règles ne peuvent pas être modifiées pour le moment.',
      rulesAppOff: (name: string) => `Activez ${name} pour modifier ses outils.`,
      rulesSignIn: 'Connectez-vous pour modifier ce que Hermes peut faire ici.',
      orgNote: (count: number) => `Votre organisation a désactivé ${count} outil${count > 1 ? 's' : ''}.`,
      orgLink: "Ouvrir l'administration des connecteurs",
      connectEnded: "La connexion n'a pas abouti.",
      connectOpenAgain: 'Rouvrir le lien',
      tokensPerCall: 'jetons par appel',
      usesPerMonth: 'utilisations sur 30 jours',
      advanced: 'Avancé',
      advancedHint: "l'entrée mcp.json et les journaux"
    },
    tools: {
      title: 'Outils',
      notInstalledBody: "Installez-le sur cet appareil pour voir les outils qu'il apporte.",
      summaryTitle: (name: string) => `Ce que Hermes peut faire avec ${name}`,
      summaryPreviewTitle: (name: string) => `Ce que Hermes pourra faire avec ${name} une fois connecté`,
      summaryCount: (count: number) => `${count} outil${count > 1 ? 's' : ''}`,
      summaryAllTools: 'Tous les outils',
      summaryOther: 'Autres',
      allToolsSwitch: 'Activer ou désactiver tous les outils',
      summaryAllOn: 'tous activés',
      summarySomeOn: (on: number, total: number) => `${on} sur ${total} activé${on > 1 ? 's' : ''}`,
      summaryOff: 'désactivés',
      showAllTools: (count: number) => (count > 1 ? `Afficher les ${count} outils` : `Afficher ${count} outil`),
      showSummary: 'Afficher le résumé',
      facetSwitch: (facet: string) => `Activer ou désactiver les outils ${facet}`,
      moreHints: (count: number) => `+${count}`,
      staleSignIn: 'Connectez-vous pour lire la dernière liste des outils.',
      searchCountPlaceholder: (count: number) => `Rechercher parmi ${count} outils`,
      toolList: (name: string) => `Outils de ${name}`,
      categorySelect: (count: number) => `${count} catégorie${count > 1 ? 's' : ''}`,
      showDeprecated: (count: number) => `Afficher ${count} obsolète${count > 1 ? 's' : ''}`,
      hideDeprecated: (count: number) => `Masquer ${count} obsolète${count > 1 ? 's' : ''}`,
      quickReadOnly: 'Lecture seule',
      quickNoDestructive: 'Désactiver les destructifs',
      quickEverythingOn: 'Tout activer',
      lockedHint: 'désactivé par votre organisation',
      turnToolOn: (tool: string) => `Activer ${tool}`,
      turnToolOff: (tool: string) => `Désactiver ${tool}`,
      showDetails: (tool: string) => `Afficher ce que fait ${tool}`,
      hideDetails: (tool: string) => `Masquer ce que fait ${tool}`,
      noMatch: 'Aucun outil ne correspond à ces filtres.',
      loading: 'Lecture de la liste des outils',
      unavailableLine: 'Liste des outils indisponible.',
      needsAuthTitle: (name: string) => `Connectez-vous à ${name} pour lire ses outils.`,
      needsAuthBody: 'La connexion reste sur cet ordinateur. Rien ne le quitte.',
      retry: 'Réessayer',
      goneTitle: (name: string) => `${name} a quitté le catalogue.`,
      goneBody: "Hermes ne peut plus l'appeler. La ligne reste jusqu'à ce que vous la supprimiez : rien ne disparaît.",
      remove: 'Supprimer',
      offTitle: (name: string) => `${name} est désactivé.`,
      offBody: "Activez-le avec l'interrupteur ci-dessus pour lire les outils qu'il apporte.",
      signedOutTitle: 'Connectez-vous à Nous pour lire la liste des outils.',
      signedOutBody: 'Vos serveurs sur cet ordinateur ne sont pas affectés.',
      conflictTitle: "Quelqu'un a modifié cette règle pendant que vous l'éditiez.",
      conflictBody: (theyOff: number, theyOn: number) => {
        const they = [
          theyOff > 0
            ? `désactivé ${theyOff} outil${theyOff > 1 ? 's' : ''} que vous avez activé${theyOff > 1 ? 's' : ''}`
            : '',
          theyOn > 0
            ? `laissé activé${theyOn > 1 ? 's' : ''} ${theyOn} outil${theyOn > 1 ? 's' : ''} que vous avez désactivé${theyOn > 1 ? 's' : ''}`
            : ''
        ].filter(Boolean)

        return `${they.length > 0 ? `Cette personne a ${they.join(' et ')}. ` : ''}Vos modifications restent à l'écran ; rien n'a été écrit.`
      },
      conflictReload: 'Recharger sa version',
      conflictSave: 'Enregistrer par-dessus sa version',
      saveFailed: "Ces règles d'outils n'ont pas été enregistrées.",
      footerDirty: (off: number, backOn: number) =>
        `${off} outil${off > 1 ? 's' : ''} désactivé${off > 1 ? 's' : ''}, ${backOn === 0 ? 'aucun' : backOn} réactivé${backOn > 1 ? 's' : ''}`,
      discard: 'Annuler les modifications',
      save: 'Enregistrer les modifications',
      saving: 'Enregistrement...'
    },
    vocabulary: {
      facetRead: {
        label: 'Lecture',
        long: 'Lit des données de cette application. Ne modifie rien.'
      },
      facetWrite: {
        label: 'Écriture',
        long: 'Crée ou modifie quelque chose dans cette application.'
      },
      facetDestructive: {
        label: 'Destructif',
        long: 'Peut supprimer définitivement quelque chose dans cette application.'
      },
      facetUnclassified: {
        label: 'Effet inconnu',
        long: "L'application n'a jamais indiqué ce que fait cet outil."
      },
      hintReadOnly: {
        label: 'Lecture seule',
        long: "L'outil déclare qu'il se contente de lire."
      },
      hintCreate: {
        label: 'Crée',
        long: 'Crée quelque chose de nouveau.'
      },
      hintUpdate: {
        label: 'Modifie',
        long: 'Modifie quelque chose qui existe déjà.'
      },
      hintDelete: {
        label: 'Supprime',
        long: 'Supprime quelque chose.'
      },
      hintDestructive: {
        label: 'Destructif',
        long: 'La modification effectuée ne peut pas être annulée ici.'
      },
      hintIdempotent: {
        label: 'Répétable',
        long: "L'exécuter deux fois produit le même effet qu'une seule fois."
      },
      hintOpenWorld: {
        label: 'Externe',
        long: 'Accède à quelque chose en dehors de cette application.'
      }
    }
  },
  sessionImport: {
    title: 'Reprendre depuis une autre application',
    subtitle: 'Importez une conversation dans Hermes et reprenez là où vous en étiez.',
    action: 'Importer une session',
    readingFrom: 'Lecture depuis',
    connectedComputer: "l'ordinateur connecté",
    destination: 'Importer dans',
    all: 'Toutes',
    search: 'Rechercher parmi les sessions chargées',
    scanning: 'Recherche des conversations',
    scanError: 'Impossible de trouver les sessions',
    scanHelp: 'Vérifiez la connexion au backend, puis réessayez. Une mise à jour du backend peut être nécessaire.',
    empty: 'Aucune conversation trouvée',
    emptyHelp: 'Les sessions Claude Code et Codex de ce backend apparaîtront ici.',
    noMatches: 'Aucune conversation correspondante',
    searchHelp: 'Essayez un autre titre ou dossier, ou chargez davantage de sessions.',
    skipped: 'Certains journaux étaient vides, illisibles ou trop volumineux pour être prévisualisés.',
    more: 'Charger davantage de sessions',
    messages: 'messages',
    choose: 'Une conversation à poursuivre',
    chooseHelp: "Choisissez une session pour lire son historique avant de l'importer dans Hermes.",
    previewLoading: "Ouverture de l'aperçu",
    previewError: 'Aperçu indisponible',
    previewHelp: 'La source a peut-être été déplacée ou modifiée. Actualisez la liste et réessayez.',
    previewLimit: 'Aperçu abrégé pour faciliter la lecture. La conversation est importée dans son intégralité.',
    you: 'Vous',
    snapshot: 'Cette conversation est déjà dans Hermes. Ouvrez votre copie existante pour continuer.',
    copyNotice:
      'Copie le texte de la conversation sans modifier les fichiers sources. Les résultats des outils et le raisonnement ne sont pas transférés.',
    importing: 'Importation…',
    open: 'Ouvrir dans Hermes',
    continue: 'Continuer dans Hermes',
    importError: "Impossible d'importer cette conversation."
  },
  common: {
    apply: 'Appliquer',
    back: 'Retour',
    save: 'Enregistrer',
    saving: 'Enregistrement…',
    cancel: 'Annuler',
    change: 'Modifier',
    choose: 'Choisir',
    clear: 'Effacer',
    close: 'Fermer',
    collapse: 'Réduire',
    confirm: 'Confirmer',
    connect: 'Connecter',
    connecting: 'Connexion',
    continue: 'Continuer',
    bots: 'Bots',
    copied: 'Copié',
    copy: 'Copier',
    copyFailed: 'Échec de la copie',
    delete: 'Supprimer',
    docs: 'Documentation',
    done: 'Terminé',
    error: 'Erreur',
    expand: 'Développer',
    failed: 'Échec',
    formatJson: 'Formater le JSON',
    free: 'Gratuit',
    loading: 'Chargement…',
    notSet: 'Non défini',
    refresh: 'Actualiser',
    remove: 'Retirer',
    replace: 'Remplacer',
    retry: 'Réessayer',
    run: 'Exécuter',
    send: 'Envoyer',
    set: 'Définir',
    skip: 'Ignorer',
    update: 'Mettre à jour',
    tryHint: term => `Essayez « ${term} »`,
    on: 'Activé',
    off: 'Désactivé'
  },
  fileMenu: {
    revealFinder: 'Afficher dans le Finder',
    revealExplorer: "Afficher dans l'Explorateur de fichiers",
    revealFileManager: 'Ouvrir le dossier parent',
    revealInSidebar: "Afficher dans l'arborescence",
    copyPath: 'Copier le chemin',
    copyRelativePath: 'Copier le chemin relatif',
    download: 'Télécharger',
    downloadSaved: 'Enregistré',
    downloadFailed: 'Échec du téléchargement',
    rename: 'Renommer…',
    delete: 'Supprimer',
    renameTitle: 'Renommer',
    renameLabel: 'Nouveau nom',
    deleteTitle: name => `Supprimer ${name} ?`,
    deleteBody: 'Il sera déplacé dans la Corbeille — vous pourrez le restaurer depuis là.',
    pathCopied: 'Chemin copié',
    revealMissing: "Ce dossier n'est pas sur cet ordinateur",
    revealUnavailable:
      "Ce chemin n'est pas sur cet ordinateur : il se trouve sur la machine du backend. Utilisez « Afficher dans l'arborescence »."
  },
  boot: {
    ready: 'Hermes Desktop est prêt',
    desktopBootFailedWithMessage: message => `Échec du démarrage : ${message}`,
    steps: {
      connectingGateway: 'Connexion au gateway desktop',
      loadingSettings: 'Chargement des paramètres Hermes',
      loadingSessions: 'Chargement des sessions récentes',
      retryingRemoteBackend: 'Reconnexion au backend Hermes distant…',
      startingDesktopConnection: 'Démarrage de la connexion desktop',
      startingHermesDesktop: 'Démarrage de Hermes Desktop…'
    },
    errors: {
      backgroundExited: "Le processus en arrière-plan de Hermes s'est arrêté.",
      backgroundExitedDuringStartup: "Le processus en arrière-plan de Hermes s'est arrêté pendant le démarrage.",
      backendStopped: 'Backend arrêté',
      restartHermes: 'Redémarrer Hermes',
      openLogs: 'Ouvrir les journaux',
      desktopBootFailed: 'Échec du démarrage',
      gatewayConnectionLost: 'Connexion au gateway perdue',
      gatewayConnectionLostDetail:
        'Nouvelle tentative en arrière-plan. Vous pouvez continuer à lire et rédiger — ouvrez les paramètres du gateway si le problème persiste.',
      reconnectNow: 'Se reconnecter maintenant',
      connectionSettings: 'Paramètres de connexion',
      gatewaySignInRequired: 'Connexion au gateway requise',
      gatewaySignInRequiredDetail:
        'Reconnectez-vous pour rétablir la connexion. Vos conversations et paramètres sont en sécurité.',
      signInAgain: 'Se reconnecter',
      ipcBridgeUnavailable: 'Le pont IPC du desktop est indisponible.'
    },
    causes: {
      exitedEarly: "Le service en arrière-plan de Hermes s'est arrêté juste après son démarrage.",
      timedOut: "Le service en arrière-plan de Hermes n'a pas répondu à temps.",
      permission: "Hermes n'a pas pu écrire dans son dossier de données (problème d'autorisation).",
      diskFull: "Le disque est plein ; Hermes n'a donc pas pu démarrer.",
      portInUse: 'Un autre programme utilise le port réseau nécessaire à Hermes.',
      installMissing:
        "Une partie de l'installation de Hermes est manquante. Choisissez Réparer l'installation pour la restaurer."
    },
    failure: {
      title: "Hermes n'a pas pu démarrer",
      description:
        "Le gateway en arrière-plan n'a pas pu se lancer. Essayez l'une des étapes de récupération ci-dessous. Rien ici ne supprime vos conversations ou paramètres.",
      details: 'Détails',
      remoteTitle: 'Connexion au gateway distante requise',
      remoteDescription:
        'Votre session de gateway distante a expiré. Connectez-vous à nouveau pour vous reconnecter. Rien ici ne supprime vos conversations ou paramètres.',
      retry: 'Réessayer',
      repairInstall: "Réparer l'installation",
      useLocalGateway: 'Utiliser le gateway local',
      gatewaySettings: 'Paramètres du gateway',
      back: 'Retour',
      openLogs: 'Ouvrir les journaux',
      repairHint: "La réparation relance l'installateur et peut prendre quelques minutes sur une machine neuve.",
      remoteSignInHint: signInLabel =>
        `Déconnecte la session navigateur distante enregistrée, puis ouvre ${signInLabel}. Utilisez le gateway local pour passer au backend intégré.`,
      signOutAndSignIn: 'Se déconnecter et se reconnecter',
      remoteFailureHint:
        "Vérifiez l'URL du gateway et la connexion dans les paramètres du gateway, ou passez au gateway local.",
      cloudDownTitle: "L'agent Nous Cloud est indisponible",
      cloudDownDescription:
        "L'agent cloud géré par Nous auquel ce gateway se connecte renvoie une erreur serveur. Il ne peut pas être redémarré depuis ici — vérifiez son état, passez au gateway local ou contactez l'assistance.",
      cloudDownHint:
        "Les boutons ci-dessous ouvrent le portail Nous, pour consulter et contrôler l'instance, ainsi que notre Discord pour obtenir de l'aide.",
      cloudDownCheckPortal: "Vérifier l'état sur le portail",
      cloudDownDiscord: "Obtenir de l'aide sur Discord",
      hideRecentLogs: 'Masquer les journaux récents',
      showRecentLogs: 'Afficher les journaux récents',
      signedInTitle: 'Connecté',
      signedInMessage: 'Reconnexion au gateway distante…',
      signInIncompleteTitle: 'Connexion incomplète',
      signInIncompleteMessage: "La fenêtre de connexion s'est fermée avant la fin de l'authentification.",
      signInFailed: 'Échec de la connexion',
      signInToRemoteGateway: 'Se connecter au gateway distante',
      signInWithProvider: provider => `Se connecter avec ${provider}`,
      identityProvider: "votre fournisseur d'identité"
    }
  },
  notifications: {
    region: 'Notifications',
    hide: 'Masquer',
    show: 'Afficher',
    more: count => `${count} autre ${count === 1 ? 'notification' : 'notifications'}`,
    clearAll: 'Tout effacer',
    dismiss: 'Fermer la notification',
    details: 'Détails',
    copyDetail: 'Copier le détail',
    copyDetailFailed: 'Impossible de copier le détail de la notification',
    backendOutOfDateTitle: 'Backend obsolète',
    backendOutOfDateMessage:
      'Votre backend Hermes est plus ancien que cette version du desktop et peut ne pas fonctionner correctement. Mettez-le à jour pour les aligner.',
    installMethodUnsupportedTitle: "Méthode d'installation non prise en charge",
    updateHermes: 'Mettre à jour Hermes',
    updateReadyTitle: 'Mise à jour prête',
    updateReadyMessage: count =>
      `${count} ${count === 1 ? 'nouvelle modification disponible' : 'nouvelles modifications disponibles'}.`,
    updateReadyMessageUnknown: 'Une nouvelle mise à jour est disponible.',
    seeWhatsNew: 'Voir les nouveautés',
    mcp: {
      needsAuthTitle: 'Le serveur MCP nécessite une nouvelle authentification',
      needsAuthMessage: name => `${name} nécessite une nouvelle authentification MCP.`,
      errorTitle: 'Serveur MCP inaccessible',
      errorMessage: name => `${name} a échoué à la vérification de l'état MCP.`,
      signIn: 'Se connecter',
      view: 'Afficher',
      disable: 'Désactiver',
      disabledMessage: name => `${name} MCP a été désactivé. Vous pouvez le réactiver depuis Capacités → MCP.`,
      disableFailed: name => `Impossible de désactiver ${name} MCP.`
    },
    errors: {
      elevenLabsNeedsKey: 'STT ElevenLabs nécessite ELEVENLABS_API_KEY.',
      elevenLabsRejectedKey: 'ElevenLabs a rejeté la clé API (401).',
      diskFull: "Le disque est plein. Libérez de l'espace disque, puis réessayez.",
      storageFailure:
        "Hermes n'a pas pu enregistrer dans son dossier de données. Ouvrez Maintenance pour le vérifier et le réparer.",
      gatewayAuthFailed: "Échec de l'authentification du gateway — vérifiez API_SERVER_KEY.",
      methodNotAllowed:
        'Le backend du desktop a rejeté cette requête (405 Method Not Allowed). Essayez de redémarrer Hermes Desktop.',
      microphonePermission: "L'autorisation du microphone a été refusée.",
      openaiRejectedApiKey: 'OpenAI a rejeté la clé API.',
      openaiTtsNeedsKey: 'TTS OpenAI nécessite VOICE_TOOLS_OPENAI_KEY ou OPENAI_API_KEY.',
      codeSkewRestartRequired:
        "Ce backend exécute encore l'ancien code après une mise à jour. Redémarrez-le pour charger le nouveau code.",
      rpcOutOfSync: "L'application et le backend ne sont pas sur la même version. Mettez-les tous les deux à jour.",
      restartHermesFailed: 'Impossible de redémarrer Hermes'
    },
    actions: {
      restartHermes: 'Redémarrer Hermes',
      openKeys: 'Ouvrir les clés',
      openGateways: 'Ouvrir les gateways',
      openMaintenance: 'Ouvrir Maintenance'
    },
    voice: {
      configureSpeechToText: 'Configurez la reconnaissance vocale pour utiliser le mode vocal.',
      couldNotStartSession: 'Impossible de démarrer la session vocale',
      microphoneAccessDenied: 'Accès au microphone refusé.',
      microphoneConstraintsUnsupported: 'Les contraintes du microphone ne sont pas prises en charge par cet appareil.',
      microphoneFailed: 'Échec du microphone',
      microphoneInUse: 'Le microphone est déjà utilisé par une autre application.',
      microphonePermissionDenied: "L'autorisation du microphone a été refusée.",
      microphoneStartFailed: "Impossible de démarrer l'enregistrement du microphone.",
      microphoneUnsupported: "Ce runtime ne prend pas en charge l'enregistrement micro.",
      noMicrophone: "Aucun microphone n'a été trouvé.",
      noSpeechDetected: 'Aucune parole détectée',
      playbackFailed: 'Échec de la lecture vocale',
      recordingFailed: "Échec de l'enregistrement vocal",
      sayStopToEnd: phrase => `Dites « ${phrase} » pour terminer la conversation vocale.`,
      transcriptionFailed: 'Échec de la transcription vocale',
      transcriptionUnavailable: "La transcription vocale n'est pas encore disponible.",
      tryRecordingAgain: 'Essayez de réenregistrer.',
      unavailable: 'Voix indisponible',
      liveEnded: 'Session vocale en direct terminée',
      liveEndedConnectionLost: 'La session vocale en direct a perdu sa connexion.',
      liveEndedClosed: 'La session vocale en direct a été fermée par le service.',
      liveError: 'Voix en direct',
      liveDelegationFailed: 'Impossible de transmettre la demande à Hermes',
      liveUnavailable: reason =>
        `Le chat vocal GPT-Live n'est pas disponible : ${reason}. Utilisation de la reconnaissance vocale à la place.`
    },
    native: {
      approvalTitle: 'Approbation requise',
      approvalTitleNamed: session => `Approbation requise — ${session}`,
      approveAction: 'Approuver',
      rejectAction: 'Rejeter',
      inputTitle: 'Saisie requise',
      inputTitleNamed: session => `Saisie requise — ${session}`,
      inputBody: 'Hermes attend votre réponse.',
      turnDoneTitle: 'Hermes a terminé',
      turnDoneBody: '',
      turnErrorTitle: 'Échec du tour',
      backgroundDoneTitle: 'Tâche en arrière-plan terminée',
      backgroundFailedTitle: 'Échec de la tâche en arrière-plan',
      creditsTitle: 'Crédits'
    }
  },
  remoteDisplayBanner: {
    message: reason =>
      `Rendu logiciel actif — affichage distant détecté (${reason}). L'accélération GPU est désactivée pour éviter les scintillements.`
  },
  billingBlock: {
    titleNous: 'Plus de crédits Nous',
    titleProvider: provider => `Plus de crédits — ${provider}`,
    fallbackMessage: 'Votre compte est à court de crédits. Ajoutez-en pour continuer.',
    openBilling: 'Ouvrir la facturation',
    addCredits: 'Ajouter des crédits',
    dismiss: 'Fermer'
  },
  sendDiagnostics: {
    title: 'Envoyer les diagnostics à Nous',
    privacyNotice:
      "Cela téléverse un paquet de débogage vers un stockage interne de Nous, et non vers un service de partage public. Il contient des informations système (système d'exploitation, versions, fournisseur et clés API configurées — jamais les clés elles-mêmes) ainsi que les journaux complets de l'agent, du gateway et du Desktop (jusqu'à 512 Ko chacun), susceptibles de contenir des conversations, des résultats d'outils et des chemins de fichiers. Les secrets sont expurgés avant l'envoi. Seuls le personnel de Nous et les modérateurs Discord autorisés peuvent consulter le paquet, qui est automatiquement supprimé après 14 jours.",
    upload: 'Envoyer',
    uploading: 'Envoi en cours…',
    cancel: 'Annuler',
    close: 'Fermer',
    copyLink: 'Copier le lien',
    uploadIdFallback: id => `Aucun lien de consultation renvoyé — indiquez l'identifiant ${id} au support`,
    doneTitle: 'Diagnostics envoyés',
    doneDescription:
      "Votre paquet a été téléversé de manière privée. Partagez le lien ci-dessous dans votre fil d'assistance afin que l'équipe puisse consulter vos journaux.",
    failedTitle: "Échec de l'envoi",
    failedHint:
      'Vous pouvez également exécuter `hermes debug share --nous` dans un terminal, ou `hermes debug share --local` pour afficher le rapport sans le téléverser.',
    handoffLead: 'Poursuivez la discussion sur :',
    links: {
      github: 'Issues GitHub',
      portal: 'Assistance du portail Nous',
      discord: 'Discord'
    }
  },
  titlebar: {
    hideSidebar: 'Masquer la barre latérale',
    showSidebar: 'Afficher la barre latérale',
    search: 'Rechercher',
    searchTitle: 'Rechercher des sessions, vues et actions',
    swapSidebarSides: 'Inverser les côtés de la barre latérale',
    hideRightSidebar: 'Masquer la barre latérale droite',
    showRightSidebar: 'Afficher la barre latérale droite',
    unreadSessions: count => (count === 1 ? '1 session non lue' : `${count} sessions non lues`),
    muteHaptics: 'Couper les vibrations',
    unmuteHaptics: 'Réactiver les vibrations',
    openSettings: 'Ouvrir les paramètres',
    openStarmap: 'Ouvrir le graphe de mémoire',
    enterHud: 'Mode HUD',
    exitHud: 'Quitter le mode HUD',
    resetHudLayout: 'Réinitialiser la taille et la position du HUD',
    layoutEditor: 'Éditeur de disposition',
    layoutEditorTitle: mod => `Éditeur de disposition — ${mod}-clic réinitialise la disposition`
  },
  keybinds: {
    title: 'Raccourcis clavier',
    subtitle: open => `Cliquez sur un raccourci pour le réaffecter · ${open} rouvre ce panneau.`,
    search: 'Rechercher des raccourcis…',
    rebind: 'Réaffecter',
    reset: 'Réinitialiser',
    resetAll: 'Tout réinitialiser',
    pressKey: 'Appuyez sur une touche…',
    set: 'défini',
    conflictWith: label => `Également affecté à « ${label} »`,
    categories: {
      composer: 'Compositeur',
      profiles: 'Profils',
      session: 'Session',
      navigation: 'Navigation',
      view: 'Affichage'
    },
    actions: {
      'keybinds.openPanel': 'Ouvrir les raccourcis clavier',
      'nav.commandPalette': 'Ouvrir la palette de commandes',
      'nav.commandCenter': 'Ouvrir le centre de commandes',
      'nav.settings': 'Ouvrir les paramètres',
      'nav.profiles': 'Ouvrir les profils',
      'nav.capabilities': 'Ouvrir les skills',
      'nav.messaging': 'Ouvrir la messagerie',
      'nav.artifacts': 'Ouvrir les artefacts',
      'nav.cron': 'Ouvrir les tâches planifiées',
      'nav.agents': 'Ouvrir les agents',
      'session.new': 'Nouvelle session',
      'session.newTab': 'Nouvel onglet de session',
      'session.newWindow': 'Nouvelle fenêtre',
      'session.next': 'Session suivante',
      'session.prev': 'Session précédente',
      'session.slot.1': 'Basculer vers la session récente 1',
      'session.slot.2': 'Basculer vers la session récente 2',
      'session.slot.3': 'Basculer vers la session récente 3',
      'session.slot.4': 'Basculer vers la session récente 4',
      'session.slot.5': 'Basculer vers la session récente 5',
      'session.slot.6': 'Basculer vers la session récente 6',
      'session.slot.7': 'Basculer vers la session récente 7',
      'session.slot.8': 'Basculer vers la session récente 8',
      'session.slot.9': 'Basculer vers la session récente 9',
      'session.focusSearch': 'Rechercher des sessions',
      'session.togglePin': 'Épingler / désépingler la session actuelle',
      'session.archive': 'Archiver la session actuelle',
      'workspace.newWorktree': 'Nouveau worktree',
      'workspace.openFolder': 'Ouvrir le dossier comme projet',
      'composer.focus': 'Mettre le focus sur le compositeur',
      'composer.modelPicker': 'Ouvrir le sélecteur de modèle',
      'composer.voice': 'Démarrer / arrêter la conversation vocale',
      'view.toggleSidebar': 'Basculer la barre latérale des sessions',
      'view.cycleSidebarGrouping': 'Changer le regroupement des sessions',
      'view.toggleRightSidebar': "Basculer l'explorateur de fichiers",
      'view.toggleReview': 'Basculer le panneau de révision',
      'view.toggleStatusbar': 'Basculer la barre de statut',
      'view.toggleTabStrip': 'Basculer les onglets',
      'view.toggleProfileRail': 'Afficher ou masquer la barre des profils',
      'view.toggleSimpleMode': 'Activer ou désactiver le mode simple',
      'view.showFiles': "Afficher l'explorateur de fichiers",
      'view.showBrowser': 'Ouvrir le navigateur',
      'view.toggleHud': 'Basculer le mode HUD',
      'hud.snapToPointer': 'Déplacer le HUD vers le pointeur (global, lorsque le HUD est ouvert)',
      'view.showTerminal': 'Basculer le terminal',
      'view.newTerminal': 'Nouveau terminal',
      'view.nextTerminal': 'Terminal suivant',
      'view.prevTerminal': 'Terminal précédent',
      'view.closeTerminal': 'Fermer le terminal',
      'view.selectionToComposer': 'Envoyer la sélection au compositeur',
      'view.terminalCopy': 'Copier la sélection du terminal',
      'view.terminalPaste': 'Coller dans le terminal',
      'view.closeTab': "Fermer l'onglet",
      'view.reopenTab': "Rouvrir l'onglet fermé",
      'view.flipPanes': 'Inverser les côtés de la barre latérale',
      'view.findInPage': 'Rechercher dans la page',
      'view.findNext': 'Rechercher la correspondance suivante',
      'view.findPrevious': 'Rechercher la correspondance précédente',
      'appearance.toggleMode': 'Basculer clair / sombre',
      'profile.default': 'Basculer vers le profil par défaut',
      'profile.switch.1': 'Basculer vers le profil 1',
      'profile.switch.2': 'Basculer vers le profil 2',
      'profile.switch.3': 'Basculer vers le profil 3',
      'profile.switch.4': 'Basculer vers le profil 4',
      'profile.switch.5': 'Basculer vers le profil 5',
      'profile.switch.6': 'Basculer vers le profil 6',
      'profile.switch.7': 'Basculer vers le profil 7',
      'profile.switch.8': 'Basculer vers le profil 8',
      'profile.switch.9': 'Basculer vers le profil 9',
      'profile.switch.10': 'Basculer vers le profil 10',
      'profile.switch.11': 'Basculer vers le profil 11',
      'profile.switch.12': 'Basculer vers le profil 12',
      'profile.switch.13': 'Basculer vers le profil 13',
      'profile.switch.14': 'Basculer vers le profil 14',
      'profile.switch.15': 'Basculer vers le profil 15',
      'profile.switch.16': 'Basculer vers le profil 16',
      'profile.switch.17': 'Basculer vers le profil 17',
      'profile.switch.18': 'Basculer vers le profil 18',
      'profile.next': 'Profil suivant',
      'profile.prev': 'Profil précédent',
      'profile.toggleAll': 'Basculer la vue tous profils',
      'profile.create': 'Créer un profil',
      'composer.send': 'Envoyer le message',
      'composer.newline': 'Insérer un saut de ligne',
      'composer.steer': 'Diriger le tour en cours',
      'composer.queue': 'Mettre le message en file',
      'composer.sendQueued': 'Envoyer le tour suivant en file',
      'composer.mention': 'Référencer des fichiers, dossiers, URLs',
      'composer.slash': 'Palette de commandes slash',
      'composer.help': 'Aide rapide',
      'composer.history': 'Parcourir le popover / historique',
      'composer.cancel': "Fermer le popover · annuler l'exécution"
    }
  },
  findInPage: {
    next: 'Correspondance suivante',
    previous: 'Correspondance précédente'
  },
  language: {
    label: 'Langue',
    description: "Choisissez la langue de l'interface du desktop.",
    saving: 'Enregistrement de la langue…',
    saveError: 'Échec de la mise à jour de la langue',
    switchTo: 'Changer de langue',
    searchPlaceholder: 'Rechercher des langues…',
    noResults: 'Aucune langue trouvée'
  },
  settings: {
    subpages: {
      appearanceTheme: 'Thème',
      appearanceTypography: 'Typographie',
      appearanceWindowLayout: 'Fenêtre et disposition',
      appearanceChatDisplay: 'Affichage des conversations',
      appearancePet: 'Compagnon',
      appearanceGeneral: 'Général',
      modelMain: 'Modèle principal',
      modelAuxiliary: 'Modèles auxiliaires',
      modelMoa: 'Mixture of Agents',
      modelFallbacks: 'Modèles de secours',
      chatBehavior: 'Comportement',
      chatAttachments: 'Pièces jointes',
      workspaceProjects: 'Projets et découverte',
      workspaceShell: 'Environnement du shell',
      workspaceFiles: 'Fichiers et exécution',
      safetyApprovals: 'Approbations',
      safetyPrivacy: 'Confidentialité et réseau',
      safetyCheckpoints: 'Points de restauration',
      browserProfile: 'Profil du navigateur',
      browserNetwork: 'URL locales et privées',
      memoryPersistent: 'Mémoire persistante',
      memoryContext: 'Contexte et compression',
      voiceConversation: 'Conversation vocale',
      voiceTranscription: 'Reconnaissance vocale',
      voiceSpeech: 'Synthèse vocale',
      advancedRuntime: "Limites de l'agent",
      advancedTools: 'Accès aux outils',
      advancedTerminal: 'Backend du terminal',
      advancedOutput: 'Limites de sortie',
      advancedDelegation: 'Sous-agents',
      advancedDesktop: 'Desktop et démarrage',
      gatewayConnection: 'Cette fenêtre',
      gatewayDevices: 'Connexions enregistrées',
      gatewayManagedUpdates: 'Mises à jour à distance',
      gatewayManagedUpdatesUnavailable:
        'Les mises à jour à distance nécessitent une version de Desktop prenant en charge les mises à jour SSH gérées.',
      gatewayManagedUpdatesEmpty:
        'Ajoutez une connexion SSH dans Connexions enregistrées pour gérer ses mises à jour ici.',
      keyboardShortcuts: 'Raccourcis clavier',
      hudGesture: 'Geste du HUD',
      screenCapture: "Capture d'écran",
      notificationAlerts: 'Alertes du bureau',
      notificationSounds: 'Sons',
      archivedSessions: 'Archives et conservation',
      defaultDirectory: 'Dossier de projet par défaut',
      vaultCredentials: 'Identifiants enregistrés',
      vaultSources: 'Gestionnaires de mots de passe',
      appUpdates: 'Version et mises à jour',
      uninstall: 'Désinstaller',
      billingOverview: "Vue d'ensemble",
      billingPlans: 'Forfaits'
    },
    closeSettings: 'Fermer les paramètres',
    exportConfig: 'Exporter la configuration',
    importConfig: 'Importer la configuration',
    resetToDefaults: 'Réinitialiser aux valeurs par défaut',
    resetConfirm: 'Réinitialiser tous les paramètres aux valeurs par défaut de Hermes ?',
    exportFailed: "Échec de l'export",
    resetFailed: 'Échec de la réinitialisation',
    nav: {
      providers: 'Fournisseurs',
      providerAccounts: 'Comptes',
      providerApiKeys: 'Clés API',
      providerCustomEndpoints: 'Points de terminaison personnalisés',
      providerLocalModels: 'Modèles locaux',
      gateway: 'Gateway',
      apiKeys: 'Outils et clés',
      keybinds: 'Raccourcis clavier',
      keysTools: 'Outils',
      keysSettings: 'Paramètres',
      mcp: 'MCP',
      archivedChats: 'Conversations archivées',
      sessions: 'Sessions',
      about: 'À propos',
      billing: 'Facturation',
      notifications: 'Notifications',
      vault: 'Mots de passe et identifiants'
    },
    plugins: {
      title: 'Plugins du desktop',
      blurb:
        "Étendez cette application, et non un agent : ces plugins sont installés une seule fois pour toute l'application, quel que soit le profil, le gateway ou la machine connectée. Les interrupteurs s'appliquent immédiatement.",
      count: n => `${n} installés`,
      openFolder: 'Ouvrir le dossier des plugins Desktop',
      rescan: 'Re-analyser',
      reveal: 'Afficher dans le gestionnaire de fichiers',
      enable: 'Activer',
      disable: 'Désactiver',
      failed: 'échec',
      empty: 'Aucun plugin desktop installé pour le moment.',
      kinds: {
        bundled: 'intégré',
        disk: 'sur le disque',
        runtime: "à l'exécution"
      },
      agentHalfMissing: 'partie agent absente ici',
      agentHalfMissingTip:
        "Il s'agit de la partie Desktop d'un plugin groupé, mais sa partie agent n'est pas installée sur le backend ou profil actuellement connecté. Installez-la depuis Capacités → Plugins.",
      installModal: {
        installFromGit: 'Installer depuis Git',
        reviewRepository: 'Examiner le dépôt',
        repoPlaceholder: 'https://github.com/proprietaire/depot',
        title: 'Installer le plugin',
        description: "Vérifiez le contenu de ce dépôt avant d'installer quoi que ce soit.",
        repoLabel: 'Dépôt',
        includesHeading: 'Ce paquet comprend',
        agentLabel: "Plugin de l'agent",
        desktopLabel: 'Interface Desktop',
        profileLabel: 'Installer pour le profil',
        agentTargetLocal: (profile, dir) => `Installe dans le backend ${profile} (${dir})`,
        agentTargetRemote: profile => `S'installe dans le backend ${profile} connecté`,
        catalogPinned: (name, sha) =>
          `Entrée « ${name} » du catalogue Hermes — le composant agent sera installé depuis le commit vérifié${sha ? ` ${sha}` : ''}, et non depuis la tête de branche.`,
        reviewedHeading: 'Entrée du catalogue vérifiée',
        reviewedIntro:
          'Cette entrée a été vérifiée manuellement à son commit épinglé. Vous pouvez encore examiner le code exact ci-dessous.',
        toolsConnected: n => (n === 1 ? '1 outil connecté' : `${n} outils connectés`),
        skillsReady: names => (names.length === 1 ? `skill ${names[0]} prêt` : `${names.length} skills prêts`),
        nextChat: "d'autres outils disponibles dans votre prochaine conversation",
        serverNotConnected: (server, reason) =>
          `Le serveur MCP ${server} n'est pas connecté${reason ? ` : ${reason}` : '.'}`,
        missingEnvAction: 'Le configurer',
        alreadyInstalled: name => `${name} est déjà installé.`,
        desktopTarget: "S'installe dans le dossier local desktop-plugins de cette application",
        desktopTargetFromPackage:
          'Chargé dans cette application depuis le paquet ci-dessus — identique pour tous les profils',
        desktopOnlyNote: "Les paquets réservés au Desktop n'installent aucun plugin dans le backend.",
        insecureWarning:
          'Cette URL utilise un protocole non sécurisé ou local. Préférez https:// ou git@ en production.',
        securityHeading: "Avant l'installation",
        securityIntro:
          'Installez uniquement des sources de confiance — consultez le dépôt ci-dessous pour vérifier ce qui sera ajouté.',
        sourceHeading: 'Code source',
        viewRepository: 'Voir le dépôt',
        viewPluginFiles: 'Voir les fichiers du plugin',
        gitCloneLabel: 'URL de clonage Git',
        enableAgent: "Activer le plugin de l'agent après l'installation",
        forceReinstall: 'Forcer la réinstallation (remplacer la version existante)',
        pinToCommit: 'Épingler à un commit (facultatif)',
        pinToCommitPlaceholder: 'SHA complet du commit sur 40 caractères',
        pinToCommitHint:
          "Toute personne installant ce SHA obtient le même code ; le plugin refusera ensuite les mises à jour jusqu'à un nouvel épinglage. Laissez vide pour utiliser le dernier commit.",
        pinToCommitInvalid:
          'Le SHA doit contenir exactement 40 caractères (les branches et tags ne sont pas acceptés).',
        install: 'Installer',
        installing: 'Installation…',
        probing: 'Inspection du dépôt…',
        probeUnavailable: "L'inspection du plugin n'est pas disponible dans cet environnement.",
        desktopUnavailable: "L'installation du plugin Desktop n'est pas disponible dans cet environnement.",
        selectComponent: 'Sélectionnez au moins un composant à installer.',
        agentSuccess: name => `Plugin de l'agent ${name} installé`,
        desktopSuccess: name => `Plugin Desktop ${name} installé`,
        agentFailed: "Échec de l'installation du plugin de l'agent",
        installUncertain:
          "Hermes n'attend plus le résultat de l'installation, mais le plugin est peut-être encore en cours d'installation. Fermez cette fenêtre et actualisez la liste des plugins avant de relancer l'installation.",
        desktopFailed: "Échec de l'installation du plugin Desktop",
        missingEnv: (name, vars) =>
          `${name} est installé, mais a besoin d'une clé pour fonctionner : ${vars}. Ajoutez-la maintenant, sinon les outils du plugin échoueront.`
      }
    },
    vault: {
      title: 'Mots de passe et identifiants',
      blurb:
        "Demandez à l'agent de se connecter à un site : il vous demandera l'identifiant la première fois, puis pourra le réutiliser. Les mots de passe sont chiffrés sur cet ordinateur et remplis directement dans la page ; le modèle ne les voit jamais.",
      count: n => `${n} enregistré${n === 1 ? '' : 's'}`,
      loadFailed: 'Impossible de charger les éléments du coffre',
      empty: 'Aucun élément enregistré',
      emptyDesc:
        "Vous n'avez rien à ajouter à l'avance. Demandez à l'agent de se connecter à un site et il vous demandera l'identifiant au moment voulu.",
      add: 'Ajouter',
      addTitle: 'Ajouter un identifiant, une carte ou une adresse',
      addDescription: "Enregistré chiffré sur cet ordinateur. L'agent ne voit jamais le mot de passe.",
      added: 'Enregistré.',
      adding: 'Enregistrement…',
      addConfirm: 'Enregistrer',
      kindField: 'Type',
      kinds: {
        login: 'Identifiant',
        payment: 'Carte bancaire',
        address: 'Adresse'
      },
      labelField: 'Libellé',
      labelPlaceholder: 'par ex. compte GitHub professionnel',
      labelRequired: 'Un libellé est requis.',
      originField: 'Origine du site',
      originPlaceholder: 'https://github.com',
      originPlaceholderCheckout: 'https://shop.example.com',
      originInvalid: 'Saisissez une URL valide, par exemple https://exemple.fr.',
      identifierTypeField: "Type d'identifiant",
      identifierTypes: {
        email: 'E-mail',
        phone: 'Téléphone',
        username: "Nom d'utilisateur"
      },
      identifierField: 'Identifiant',
      identifierShown: identifier => identifier,
      passwordField: 'Mot de passe',
      loginFieldsRequired: "L'identifiant et le mot de passe sont requis.",
      cardNumberField: 'Numéro de carte',
      cardNameField: 'Nom sur la carte',
      expMonthField: "Mois d'expiration",
      expYearField: "Année d'expiration",
      cvcField: 'CVC',
      postalField: 'Code postal',
      addressLine1Field: 'Adresse ligne 1',
      addressLine2Field: 'Adresse ligne 2',
      cityField: 'Ville',
      stateField: 'État / région',
      countryField: 'Pays',
      optional: '(facultatif)',
      createdOn: date => `Ajouté ${date}`,
      deleteAction: "Supprimer l'élément enregistré",
      otpField: "Clé d'authentificateur",
      otpPlaceholder: 'Secret Base32 ou lien otpauth://',
      otpHint:
        "La « clé d'installation » que le site affiche lorsque vous activez 2FA. Avec elle enregistrée, Hermes génère lui-même les codes.",
      twoFactorBadge: '2FA auto',
      deleteTitle: 'Supprimer cet élément ?',
      deleteDescription: label => `« ${label} » sera supprimé définitivement.`,
      deleteConfirm: 'Supprimer',
      sources: {
        title: 'Gestionnaires de mots de passe',
        blurb:
          "Les gestionnaires de mots de passe installés sont repérés automatiquement. L'agent vous demande de déverrouiller l'un la première fois qu'il en a besoin (une fois par session) ; seul un jeton de session reste en mémoire, et l'agent ne voit jamais votre mot de passe principal ou aucun identifiant.",
        toggleFailed: 'Impossible de mettre à jour le gestionnaire de mots de passe',
        notInstalled: name =>
          `Non détecté. Installez l’outil en ligne de commande ${name} et connectez-vous ; Hermes le repère automatiquement.`,
        disabledDesc: 'Détecté mais désactivé pour Hermes.',
        lockedDesc:
          "Détecté. L'agent vous demandera de le déverrouiller lorsqu'il en a besoin, ou déverrouillez maintenant.",
        unlockedDesc:
          "Déverrouillé pour cette session. Se verrouille automatiquement après 30 minutes d'inactivité ou lorsque Hermes se ferme.",
        statusLocked: 'Verrouillé',
        statusNotDetected: 'Non détecté',
        statusOff: 'Désactivé',
        statusUnlocked: 'Déverrouillé',
        unlock: 'Déverrouiller',
        unlocking: 'Déverrouillage…',
        lock: 'Verrouiller',
        unlocked: name => `${name} est déverrouillé pour cette session.`,
        unlockTitle: name => `Déverrouiller ${name}`,
        unlockDescription:
          "Entrez votre mot de passe principal. Il est transmis au gestionnaire de mots de passe sur cette machine et jeté — il n'est jamais stocké, enregistré, ni montré à l'agent.",
        masterPasswordPlaceholder: 'Mot de passe principal'
      }
    },
    notifications: {
      title: 'Notifications',
      intro: "Notifications système (pas les toasts de l'application). Par appareil.",
      enableAll: 'Activer les notifications',
      enableAllDesc: 'Désactivé coupe toutes les notifications ci-dessous.',
      focusedHint: 'Les alertes de fin ne se déclenchent que quand Hermes est en arrière-plan.',
      kinds: {
        approval: {
          label: 'Approbation requise',
          description: "Une commande attend que vous l'approuviez ou la rejetiez."
        },
        input: {
          label: 'Saisie requise',
          description: "Hermes a posé une question ou a besoin d'un mot de passe ou d'un secret."
        },
        turnDone: {
          label: 'Réponse prête',
          description: "Un tour s'est terminé pendant que Hermes était en arrière-plan."
        },
        turnError: {
          label: 'Échec du tour',
          description: 'Erreurs de tour en arrière-plan.'
        },
        backgroundDone: {
          label: 'Tâche en arrière-plan terminée',
          description: "Une commande terminal en arrière-plan s'est terminée."
        },
        credits: {
          label: 'Alertes de crédits',
          description: "L'accès aux crédits est suspendu ou rétabli."
        },
        plugin: {
          label: 'Notifications des plugins',
          description: 'Un plugin desktop a envoyé une notification pendant que Hermes était en arrière-plan.'
        }
      },
      test: 'Envoyer une notification de test',
      testTitle: 'Hermes',
      testBody: 'Les notifications fonctionnent.',
      testSent:
        "Test envoyé. Si rien n'apparaît, vérifiez les autorisations de notification de votre système et le mode Ne pas déranger.",
      testUnsupported: 'Ce système ne prend pas en charge les notifications natives.',
      completionSoundTitle: 'Son de fin',
      completionSoundDesc: "Se joue à la fin d'un tour d'agent. Choisissez un préréglage et prévisualisez-le ici.",
      completionSoundPreview: 'Aperçu'
    },
    sections: {
      model: 'Modèle',
      chat: 'Conversation',
      appearance: 'Apparence',
      workspace: 'Espace de travail',
      safety: 'Sécurité',
      memory: 'Mémoire et contexte',
      voice: 'Voix',
      advanced: 'Avancé'
    },
    searchPlaceholder: {
      about: 'À propos de Hermes Desktop',
      config: 'Rechercher dans les paramètres...',
      gateway: 'Connexion au gateway...',
      keys: 'Rechercher des clés API...',
      mcp: 'Rechercher des serveurs MCP...',
      sessions: 'Rechercher des sessions archivées...'
    },
    modeOptions: {
      light: {
        label: 'Clair',
        description: 'Surfaces lumineuses'
      },
      dark: {
        label: 'Sombre',
        description: 'Espace de travail sans éblouissement'
      },
      system: {
        label: 'Système',
        description: "Suivre l'apparence du système"
      }
    },
    appearance: {
      title: 'Apparence',
      intro:
        'Exclusif au desktop. Le mode contrôle la luminosité ; le thème contrôle la palette et le chrome de la conversation.',
      colorMode: 'Mode couleur',
      colorModeDesc: 'Choisissez un mode fixe ou laissez Hermes suivre le paramètre système.',
      toolViewTitle: "Affichage des appels d'outil",
      toolViewDesc:
        'Le mode Produit masque les charges utiles brutes ; le mode Technique affiche les entrées/sorties complètes.',
      hideCodeDiffsTitle: 'Masquer les diffs de code',
      hideCodeDiffsDesc:
        'Afficher les modifications de fichiers sous forme de lignes d’outil avec le nombre de lignes ajoutées/supprimées, sans le code.',
      hideThreadTimelineTitle: 'Masquer les barres de chronologie',
      hideThreadTimelineDesc: 'Masquer les barres de navigation le long du bord droit de chaque conversation.',
      reasoningCollapsedTitle: 'Réduire le raisonnement par défaut',
      reasoningCollapsedDesc:
        "Conserver le raisonnement diffusé en continu sans le développer tant que vous ne l'ouvrez pas.",
      uiScaleTitle: "Échelle de l'interface",
      uiScaleDesc: (percent: number) =>
        `Redimensionne le texte et les contrôles dans toute l'application. Cmd/Ctrl avec +, - et 0 fonctionne aussi. Actuel : ${percent}%.`,
      sessionDensityTitle: 'Densité de la liste des sessions',
      sessionDensityDesc:
        'Choisissez la quantité de contexte affichée sous les titres de session dans la barre latérale.',
      sessionDensityCompact: 'Compacte',
      sessionDensityComfortable: 'Confortable',
      sessionDensityDetailed: 'Détaillée',
      tabStripTitle: "Barre d'onglets",
      tabStripDesc:
        "Affiche les onglets au-dessus d'une zone. Le mode automatique les masque lorsqu'une zone ne contient qu'un seul panneau.",
      tabStripAuto: 'Automatique',
      tabStripAlways: 'Toujours',
      tabStripNever: 'Jamais',
      appActionsTitle: "Actions de l'application",
      appActionsDesc:
        'Choisissez le côté de la barre de titre où placer Paramètres, Disposition et HUD. Le côté droit laisse de la place aux onglets à gauche.',
      appActionsLeft: 'À gauche',
      appActionsRight: 'À droite',
      terminalFontTitle: 'Police du terminal',
      terminalFontDesc:
        'Choisissez une police installée pour les terminaux Desktop. Les Nerd Fonts affichent correctement Powerlevel10k et les icônes du shell ; laissez vide pour utiliser JetBrains Mono intégré.',
      terminalFontPlaceholder: 'MesloLGS NF ou une pile de polices CSS',
      terminalFontPreview: 'Aperçu des glyphes',
      terminalFontReset: 'Utiliser la valeur par défaut',
      chatFontTitle: 'Police de la conversation',
      chatFontDesc:
        "Choisissez une police installée pour la conversation et le reste de l'application. Pratique pour des polices plus lisibles comme OpenDyslexic ; laissez vide pour utiliser celle du thème.",
      chatFontPlaceholder: 'OpenDyslexic ou une pile de polices CSS',
      chatFontPreview: 'Aperçu',
      chatFontSample: 'Portez ce vieux whisky au juge blond qui fume. 0123456789',
      chatFontReset: 'Utiliser la police du thème',
      translucencyTitle: 'Translucidité de la fenêtre',
      translucencyDesc: 'Voir votre bureau à travers toute la fenêtre. macOS et Windows uniquement.',
      translucencyGlassDesc:
        'Verre mat : le bureau reste visible à travers un flou uniforme tandis que le texte demeure net. macOS uniquement.',
      translucencyModeClear: 'Transparent',
      translucencyModeGlass: 'Verre',
      translucencyTintTitle: 'Teinte',
      translucencyFadeTitle: 'Atténuation',
      translucencyFrostTitle: 'Givre',
      translucencyFrost: {
        'under-window': 'Profond',
        popover: 'Doux',
        titlebar: 'Clair',
        header: 'Éclat'
      },
      translucencyScopeTitle: 'Zone',
      translucencyScope: {
        window: 'Fenêtre entière',
        sidebar: 'Barre latérale uniquement'
      },
      backdropTitle: 'Arrière-plan de la conversation',
      backdropDesc: "L'image de statue discrète derrière la conversation.",
      userBubbleTitle: 'Bulle des messages',
      userBubbleDesc: 'Transparence de vos messages : fond opaque à 0 ; seul le contour reste visible à 100.',
      textDirectionTitle: 'Sens du texte',
      textDirectionDesc:
        "Sens d'écriture des messages et du champ de saisie. Auto suit la première lettre de chaque paragraphe ; choisissez un sens quand un texte mixte s'aligne mal. Le code reste toujours de gauche à droite.",
      textDirection: { auto: 'Auto', rtl: 'De droite à gauche', ltr: 'De gauche à droite' },
      introSplashTitle: "Écran d'accueil",
      introSplashDesc: "Le logo et l'invite affichés dans une conversation vide.",
      reactionsTitle: 'Réactions aux messages',
      reactionsDesc: 'Réactions emoji façon iMessage — réagissez aux messages, et Hermes peut réagir aux vôtres.',
      tipsTitle: "Astuces dans l'application",
      tipsDesc:
        "Une petite bulle désigne occasionnellement une partie de l'application lorsque vous êtes inactif ou lorsque Hermes peut vous aider. Fermer une astuce la masque définitivement.",
      tipsReset: (count: number) =>
        `Réafficher ${count} astuce${count === 1 ? '' : 's'} fermée${count === 1 ? '' : 's'}`,
      toursTitle: 'Visites guidées',
      toursDesc:
        "Laissez Hermes vous guider dans l'application en assombrissant l'écran et en mettant chaque étape en évidence.",
      composerPopoutTitle: 'Détacher la zone de saisie',
      composerPopoutDesc: "Autoriser la zone de saisie à s'ouvrir dans une fenêtre flottante distincte.",
      vibeHeartsTitle: "Cœurs d'ambiance",
      vibeHeartsDesc:
        "Des cœurs flottants apparaissent lorsque vous dites merci, « je t'aime », « good bot » ou envoyez un cœur. Cette option est indépendante des réactions aux messages ci-dessus.",
      embedsTitle: 'Intégrations en ligne',
      embedsDesc:
        "Les aperçus enrichis se chargent depuis des sites tiers (YouTube, X, …). Demander affiche un espace réservé jusqu'à ce que vous autorisiez chacun ; Toujours les charge automatiquement ; Désactivé conserve les liens simples.",
      embedsAsk: 'Demander',
      embedsAlways: 'Toujours',
      embedsOff: 'Désactivé',
      embedsReset: (count: number) =>
        `Réinitialiser ${count} ${count === 1 ? 'service autorisé' : 'services autorisés'}`,
      resumeLastSessionTitle: 'Rouvrir la dernière conversation au lancement',
      resumeLastSessionDesc:
        "Lorsque cette option est activée, l'application rouvre votre conversation la plus récente après un démarrage à froid. Désactivez-la pour toujours commencer par une nouvelle conversation.",
      product: 'Produit',
      productDesc: 'Activité des outils lisible avec des résumés concis.',
      technical: 'Technique',
      technicalDesc: 'Inclure les arguments/résultats bruts des outils et les détails de bas niveau.',
      themeTitle: 'Thème',
      themeDesc: "Palettes desktop uniquement. Le mode sélectionné s'applique par-dessus.",
      themeSearchPlaceholder: 'Rechercher dans vos thèmes ou sur le Marketplace VS Code…',
      themeProfileNote: profile => `Enregistré pour le profil ${profile} — chaque profil conserve son propre thème.`,
      installTitle: 'Installer depuis VS Code',
      installDesc:
        "Collez un identifiant d'extension Marketplace (ex. dracula-theme.theme-dracula) pour convertir son thème de couleur en palette desktop.",
      installPlaceholder: 'éditeur.extension',
      installButton: 'Installer',
      installing: 'Installation…',
      installError: "Impossible d'installer ce thème.",
      installed: name => `« ${name} » installé.`,
      removeTheme: 'Supprimer le thème',
      importedBadge: 'Importé',
      pet: {
        title: 'Animal de compagnie',
        intro:
          "Adoptez une mascotte petdex animée qui flotte au-dessus de l'application et réagit aux actions de Hermes — court pendant l'exécution des outils, fête les réussites, boude les erreurs.",
        restartHint:
          "Les animaux de compagnie nécessitent un redémarrage rapide — l'application en cours a démarré avant l'ajout de cette fonctionnalité. Fermez et rouvrez Hermes, puis revenez ici.",
        scaleTitle: 'Taille',
        scaleDesc: "Redimensionnez la mascotte flottante. S'applique partout instantanément.",
        roamTitle: 'Errer',
        roamDesc: "Laissez l'animal errer dans la fenêtre pendant les moments d'inactivité.",
        chooseTitle: 'Choisir un animal',
        chooseDesc: "Le choisir l'installe (si nécessaire) et l'active.",
        searchPlaceholder: 'Rechercher des animaux…',
        unreachable: "Impossible d'atteindre la galerie petdex. Vérifiez votre connexion et rouvrez cette page.",
        noMatch: query => `Aucun animal ne correspond à « ${query} ».`,
        installedTag: 'installé',
        generatedTag: 'Généré',
        countCapped: (cap, total) => `Affichage de ${cap} sur ${total} — tapez pour affiner.`,
        count: n => `${n} animal${n === 1 ? '' : 's'}.`,
        uninstall: name => `Désinstaller ${name}`,
        delete: name => `Supprimer ${name}`,
        deleteTitle: name => `Supprimer ${name} ?`,
        deleteBody: "Cela supprime définitivement l'animal — il ne peut pas être réinstallé.",
        deleteConfirm: 'Supprimer',
        rename: name => `Renommer ${name}`,
        renameTitle: "Renommer l'animal",
        renamePlaceholder: 'Nommez votre animal',
        renameSave: 'Enregistrer',
        exportPet: name => `Exporter ${name}`,
        adoptFailed: slug => `Impossible d'adopter ${slug}`,
        uninstallFailed: slug => `Impossible de désinstaller ${slug}`,
        renameFailed: slug => `Impossible de renommer ${slug}`,
        exportFailed: slug => `Impossible d'exporter ${slug}`,
        noneAvailable: 'Aucun animal disponible pour le moment.',
        turnOnFailed: "Impossible d'activer l'animal.",
        turnOffFailed: "Impossible de désactiver l'animal."
      }
    },
    fieldLabels: defineFieldCopy({
      model: 'Modèle par défaut',
      modelContextLength: 'Fenêtre de contexte du modèle principal (forçage)',
      fallbackProviders: 'Modèles de secours',
      toolsets: "Ensembles d'outils activés",
      timezone: 'Fuseau horaire',
      display: {
        personality: 'Personnalité',
        showReasoning: 'Blocs de raisonnement'
      },
      desktop: {
        repoScanEnabled: 'Découverte automatique de dépôts',
        repoScanRoots: 'Racines de découverte de dépôts',
        repoScanExcludePaths: 'Chemins de dépôt exclus'
      },
      agent: {
        maxTurns: "Étapes max. de l'agent",
        imageInputMode: "Pièces jointes d'image",
        apiMaxRetries: 'Nouveaux essais API',
        serviceTier: 'Niveau de service',
        toolUseEnforcement: "Contrôle de l'utilisation des outils"
      },
      terminal: {
        cwd: 'Répertoire de travail',
        backend: "Backend d'exécution",
        timeout: "Délai d'expiration de la commande",
        persistentShell: 'Shell persistant',
        envPassthrough: "Transmission de l'environnement",
        dockerImage: 'Image Docker',
        singularityImage: 'Image Singularity',
        modalImage: 'Image Modal',
        daytonaImage: 'Image Daytona'
      },
      fileReadMaxChars: 'Limite de lecture de fichier',
      toolOutput: {
        maxBytes: 'Limite de sortie du terminal',
        maxLines: 'Limite de pagination de fichier',
        maxLineLength: 'Limite de longueur de ligne'
      },
      codeExecution: {
        mode: "Mode d'exécution du code"
      },
      approvals: {
        mode: "Mode d'approbation",
        timeout: "Délai d'expiration de l'approbation",
        mcpReloadConfirm: 'Confirmer les rechargements MCP'
      },
      commandAllowlist: "Liste d'autorisation des commandes",
      security: {
        redactSecrets: 'Masquer les secrets',
        allowPrivateUrls: 'Autoriser les adresses web privées'
      },
      browser: {
        allowPrivateUrls: 'Adresses web privées du navigateur',
        autoLocalForPrivateUrls: 'Navigateur local pour les adresses web privées',
        useRealProfile: 'Utiliser mon vrai profil de navigateur'
      },
      checkpoints: {
        enabled: 'Points de contrôle de fichiers',
        maxSnapshots: 'Limite de points de contrôle'
      },
      voice: {
        maxRecordingSeconds: "Durée maximale d'enregistrement",
        autoTts: 'Lire les réponses à haute voix',
        voiceChatMode: 'Mode de conversation vocale',
        gptLive: {
          voice: 'Voix GPT-Live',
          instructions: 'Personnalité GPT-Live'
        }
      },
      stt: {
        enabled: 'Reconnaissance vocale',
        echoTranscripts: 'Réémettre les transcriptions',
        provider: 'Fournisseur de reconnaissance vocale',
        local: {
          model: 'Modèle de transcription local',
          language: 'Langue de transcription'
        },
        openai: {
          model: 'Modèle STT OpenAI'
        },
        groq: {
          model: 'Modèle STT Groq'
        },
        mistral: {
          model: 'Modèle STT Mistral'
        },
        elevenlabs: {
          modelId: 'Modèle STT ElevenLabs',
          languageCode: 'Langue ElevenLabs',
          tagAudioEvents: 'Étiqueter les événements audio',
          diarize: 'Diarisation des locuteurs'
        }
      },
      tts: {
        provider: 'Fournisseur de synthèse vocale',
        edge: {
          voice: 'Voix Edge'
        },
        openai: {
          model: 'Modèle TTS OpenAI',
          voice: 'Voix OpenAI'
        },
        elevenlabs: {
          voiceId: 'Voix ElevenLabs',
          modelId: 'Modèle ElevenLabs'
        },
        xai: {
          voiceId: 'Voix xAI (Grok)',
          language: 'Langue xAI',
          speed: 'Vitesse de lecture xAI',
          autoSpeechTags: 'Étiquettes vocales automatiques xAI',
          optimizeStreamingLatency: 'Optimisation de la latence streaming xAI',
          sampleRate: "Fréquence d'échantillonnage xAI",
          bitRate: 'Débit binaire xAI'
        },
        minimax: {
          model: 'Modèle TTS MiniMax',
          voiceId: 'Voix MiniMax'
        },
        mistral: {
          model: 'Modèle TTS Mistral',
          voiceId: 'Voix Mistral'
        },
        gemini: {
          model: 'Modèle TTS Gemini',
          voice: 'Voix Gemini'
        },
        neutts: {
          model: 'Modèle NeuTTS',
          device: 'Appareil NeuTTS'
        },
        kittentts: {
          model: 'Modèle KittenTTS',
          voice: 'Voix KittenTTS'
        },
        piper: {
          voice: 'Voix Piper'
        },
        deepinfra: {
          model: 'Modèle TTS DeepInfra',
          voice: 'Voix DeepInfra'
        }
      },
      memory: {
        memoryEnabled: 'Mémoire persistante',
        userProfileEnabled: 'Profil utilisateur',
        memoryCharLimit: 'Budget mémoire',
        userCharLimit: 'Budget profil',
        provider: 'Fournisseur de mémoire'
      },
      context: {
        engine: 'Moteur de contexte'
      },
      compression: {
        enabled: 'Auto-compression',
        threshold: 'Seuil de compression',
        codexGpt55Autoraise: 'Relèvement automatique de la compression Codex',
        targetRatio: 'Objectif de compression',
        protectLastN: 'Messages récents protégés'
      },
      auxiliary: {
        compression: {
          timeout: 'Délai du modèle de compression (s)'
        }
      },
      delegation: {
        model: 'Modèle sous-agent',
        provider: 'Fournisseur sous-agent',
        maxIterations: 'Limite de tours sous-agent',
        maxConcurrentChildren: 'Sous-agents parallèles',
        childTimeoutSeconds: 'Délai sous-agent',
        reasoningEffort: 'Intensité du raisonnement sous-agent'
      },
      updates: {
        nonInteractiveLocalChanges: 'Modifications locales à la mise à jour intégrée'
      }
    }),
    fieldDescriptions: defineFieldCopy({
      model: 'Utilisé pour les nouvelles conversations sauf si vous choisissez un autre modèle dans le compositeur.',
      modelContextLength:
        'Remplace la fenêtre de contexte détectée du modèle de conversation PRINCIPAL uniquement (en jetons). Laissez 0 pour utiliser la valeur détectée du modèle sélectionné. N’affecte pas les modèles auxiliaires/MoA.',
      fallbackProviders: 'Entrées fournisseur:modèle de secours à essayer si le modèle par défaut échoue.',
      display: {
        personality: "Style par défaut de l'assistant pour les nouvelles sessions.",
        showReasoning: 'Afficher les sections de raisonnement quand le backend les fournit.'
      },
      desktop: {
        repoScanEnabled: 'Analyser les dossiers locaux à la recherche de dépôts Git à afficher dans Projets.',
        repoScanRoots: 'Dossiers à analyser. Laissez vide pour analyser votre répertoire personnel.',
        repoScanExcludePaths: 'Dossiers et leurs descendants à ignorer lors de la découverte de dépôts.'
      },
      timezone: 'Identifiant de fuseau horaire IANA. Si vide, utilise le fuseau horaire du système.',
      browser: {
        useRealProfile:
          'La navigation locale utilise vos vraies connexions. Hermes copie le profil de votre navigateur par défaut (cookies, connexions, préférences) dans un instantané géré et le pilote avec son Chromium intégré — votre profil actif n’est jamais ouvert directement, et la copie est actualisée à chaque exécution. Permet aussi à l’agent d’ouvrir sur demande une session locale avec votre vrai profil, même si un backend de navigateur cloud est configuré. Seuls les navigateurs Chromium (Chrome, Edge, Brave, Brave Origin, Chromium) sont pris en charge ; un navigateur par défaut non Chromium échoue avec un message clair. Désactivé par défaut.'
      },
      agent: {
        imageInputMode: 'Contrôle la façon dont les pièces jointes image sont envoyées au modèle.',
        maxTurns: "Limite supérieure de tours d'appel d'outils avant que Hermes n'arrête une exécution."
      },
      terminal: {
        cwd: 'Dossier de projet par défaut pour les outils et le terminal.',
        persistentShell: "Conserve l'état du shell entre les commandes lorsque le backend le permet.",
        envPassthrough: "Variables d'environnement à transmettre à l'exécution des outils.",
        dockerImage: "Image de conteneur utilisée lorsque le backend d'exécution est Docker.",
        singularityImage: "Image utilisée lorsque le backend d'exécution est Singularity.",
        modalImage: "Image utilisée lorsque le backend d'exécution est Modal.",
        daytonaImage: "Image utilisée lorsque le backend d'exécution est Daytona."
      },
      codeExecution: {
        mode: "Degré de restriction de l'exécution du code au projet actuel."
      },
      fileReadMaxChars: 'Nombre maximal de caractères que Hermes peut lire dans une demande de fichier.',
      approvals: {
        mode: 'Comment Hermes gère les commandes nécessitant une approbation explicite.',
        timeout: "Durée d'attente des invites d'approbation avant expiration."
      },
      security: {
        redactSecrets: "Masque les secrets détectés du contenu visible par le modèle lorsque c'est possible."
      },
      checkpoints: {
        enabled: 'Créer des instantanés de restauration avant les modifications de fichiers.'
      },
      memory: {
        memoryEnabled: 'Sauvegarder des mémoires durables pouvant aider les sessions futures.',
        userProfileEnabled: 'Maintenir un profil compact des préférences utilisateur.'
      },
      context: {
        engine: 'Stratégie pour gérer les longues conversations proches de la limite de contexte.'
      },
      compression: {
        enabled: 'Résumer le contexte ancien quand les conversations grossissent.',
        codexGpt55Autoraise: 'Relever la compression à 85 % pour les modèles ChatGPT Codex OAuth pris en charge.'
      },
      auxiliary: {
        compression: {
          timeout:
            'Secondes d’attente du modèle de compression auxiliaire par appel (120 par défaut). Augmentez pour les modèles locaux lents.'
        }
      },
      voice: {
        autoTts: "Lit automatiquement les réponses de l'assistant à voix haute.",
        voiceChatMode:
          'chained : reconnaissance vocale → Hermes → synthèse vocale avec les fournisseurs ci-dessous. gpt-live : un modèle vocal OpenAI full-duplex (gpt-live-1) écoute et parle, et confie chaque vraie demande à Hermes — le modèle que vous avez sélectionné répond avec l’ensemble des outils. Nécessite une clé API OpenAI ; la couche vocale est facturée 0,05 $ par minute.',
        gptLive: {
          voice: 'Voix du mode GPT-Live. Les identifiants de voix personnalisés sont acceptés.',
          instructions:
            'Phrases supplémentaires pour la personnalité vocale en direct (ton, rythme, langue). Hermes conserve son propre prompt système.'
        }
      },
      tts: {
        xai: {
          voiceId: 'ID de voix xAI (ex. eve) ou un ID de voix personnalisé.',
          language: 'Code de langue parlée (ex. en, pt-BR) ou "auto" pour la détection automatique.',
          speed: 'Vitesse de lecture. 0.7 = plus lent, 1.0 = normal, 1.5 = plus rapide.',
          autoSpeechTags:
            'Laisser un LLM insérer des balises audio expressives ([laughing], [sighs]) dans le script avant la synthèse.',
          optimizeStreamingLatency: 'Compromis latence/qualité. 0 = meilleure qualité, 2 = latence minimale.',
          sampleRate:
            "Fréquence d'échantillonnage audio en Hz. Plus élevée = meilleure qualité, fichiers plus volumineux.",
          bitRate: "Débit binaire MP3 en bps. S'applique uniquement lorsque le codec est mp3."
        },
        neutts: {
          device: "Périphérique d'inférence local pour NeuTTS."
        }
      },
      stt: {
        enabled: 'Activer la transcription vocale locale ou via fournisseur.',
        echoTranscripts: 'Publier la transcription brute 🎙️ des messages vocaux dans la conversation.',
        elevenlabs: {
          languageCode: 'Code de langue ISO-639-3 facultatif. Vide = détection automatique par ElevenLabs.'
        }
      },
      updates: {
        nonInteractiveLocalChanges:
          "Quand Hermes se met à jour depuis l'application (sans invite de terminal), conserve les modifications locales (stash) ou les abandonne (discard). Les mises à jour via le terminal demandent toujours confirmation."
      }
    }),
    uninstallSection: {
      dangerZone: 'Zone dangereuse',
      checkingInstalled: 'Vérification des éléments installés…',
      uninstallHermes: 'Désinstaller Hermes',
      chooseHowMuch:
        'Choisissez ce que vous souhaitez supprimer. L’application se ferme pour terminer ; rouvrez le programme d’installation à tout moment pour revenir.',
      confirmUninstall: 'Confirmer la désinstallation',
      confirmBody: what => `Cette action supprime ${what}. Elle est irréversible.`,
      appLabel: 'Application :',
      couldNotStart: 'La désinstallation n’a pas pu démarrer.',
      uninstalling: 'Désinstallation…',
      yesUninstall: 'Oui, désinstaller',
      options: {
        gui: {
          title: 'Désinstaller uniquement l’interface de chat',
          description:
            'Supprime cette application de bureau. L’agent Hermes, votre configuration et vos conversations sont conservés.',
          consequence: 'l’interface de chat de bureau (cette application et ses données)'
        },
        lite: {
          title: 'Désinstaller l’interface et l’agent, conserver mes données',
          description:
            'Supprime l’application et l’agent Hermes, mais conserve la configuration, les conversations et les secrets pour une future réinstallation.',
          consequence:
            'l’interface de chat et l’agent Hermes (la configuration, les conversations et les secrets sont conservés)'
        },
        full: {
          title: 'Tout désinstaller',
          description:
            'Supprime l’application, l’agent et toutes les données utilisateur : configuration, conversations, tâches planifiées, secrets, journaux.',
          consequence:
            'TOUT — l’interface de chat, l’agent Hermes et l’ensemble de votre configuration, de vos conversations, secrets et journaux'
        }
      }
    },
    poolLimits: {
      warmBotBackendsAria: 'Backends de bots maintenus actifs',
      warmBotBackendsTitle: 'Backends de bots actifs',
      backendIdleTimeoutAria: "Délai d'inactivité du backend en millisecondes",
      backendIdleTimeoutTitle: "Délai d'inactivité du backend"
    },
    customEndpoints: {
      active: 'Actif',
      apiKeySet: 'Clé API définie',
      use: 'Utiliser',
      editTitle: 'Modifier le point de terminaison',
      addTitle: 'Ajouter un point de terminaison',
      fields: {
        name: 'Nom',
        providerId: 'ID du fournisseur',
        endpointUrl: 'URL du point de terminaison',
        defaultModel: 'Modèle par défaut',
        context: 'Contexte',
        apiKey: 'Clé API',
        apiKeyNewPlaceholder: 'Laissez vide pour conserver la clé actuelle',
        apiKeyPlaceholder: 'Facultatif',
        useNewChats: 'Utiliser pour les nouvelles conversations',
        discoverModels: 'Découvrir les modèles'
      },
      test: 'Tester',
      save: 'Enregistrer',
      newEndpoint: 'Nouveau point de terminaison',
      apiMode: 'Mode API',
      autoDetect: 'Détection automatique',
      couldNotLoad: 'Impossible de charger les points de terminaison personnalisés',
      endpointSaved: 'Point de terminaison personnalisé enregistré.',
      saveFailed: 'Échec de l’enregistrement',
      endpointReachable: 'Le point de terminaison est joignable.',
      endpointReachableTransport: transport => `Le point de terminaison est joignable (route ${transport} servie).`,
      endpointReachableModels: (reachable, count) =>
        `${reachable} ${count} modèle${count > 1 ? 's' : ''} trouvé${count > 1 ? 's' : ''}.`,
      endpointValidationFailed: 'La validation du point de terminaison a échoué.',
      validationFailed: 'Échec de la validation',
      activationFailed: 'Échec de l’activation',
      deleteConfirm: name => `Supprimer ${name} ?`,
      deleteFailed: 'Échec de la suppression',
      title: 'Points de terminaison personnalisés',
      deleteEndpoint: 'Supprimer le point de terminaison',
      emptyDescription: 'Ajoutez ci-dessous un point de terminaison compatible OpenAI.',
      emptyTitle: 'Aucun point de terminaison personnalisé',
      namePlaceholder: 'Proxy Axet',
      contextPlaceholder: 'Auto'
    },
    computerUse: {
      accessibility: 'Accessibilité',
      screenRecording: "Enregistrement de l'écran",
      driverHealth: 'État du pilote'
    },
    about: {
      updates: 'Mises à jour'
    },
    config: {
      minimizeToTrayTitle: 'Réduire dans la barre d’état',
      minimizeToTrayDesc:
        'Réduire les fenêtres ou fermer la fenêtre principale les masque dans la zone de notification (barre des menus sur macOS) et Hermes continue de s’exécuter. Utilisez Quitter Hermes dans le menu de la zone de notification ou Cmd+Q pour quitter. Désactivé par défaut ; s’applique uniquement à cet appareil.',
      minimizeToTrayUnavailable:
        'La zone de notification est indisponible. Les fenêtres seront réduites et fermées normalement. Désactivez puis réactivez cette option pour réessayer.',
      none: 'Aucun',
      noneParen: '(aucun)',
      builtinOnly: 'Intégré uniquement',
      notSet: 'Non défini',
      commaSeparated: 'valeurs séparées par des virgules',
      searchPlaceholder: 'Rechercher…',
      noResults: 'Aucun résultat trouvé',
      systemDefault: 'Par défaut du système',
      loading: 'Chargement de la configuration Hermes...',
      emptyTitle: 'Rien à configurer',
      emptyDesc: 'Cette section ne contient aucun paramètre ajustable.',
      failedLoad: 'Échec du chargement des paramètres',
      autosaveFailed: "Échec de l'enregistrement automatique",
      imported: 'Configuration importée',
      invalidJson: 'JSON de configuration invalide',
      toolsetsWipeConfirm:
        "Retirer tous les ensembles d'outils activés ? Cela désactive la mémoire, le terminal, la recherche web, la délégation et la plupart des autres outils jusqu'à leur réactivation.",
      keepAwakeTitle: "Garder l'ordinateur éveillé",
      keepAwakeDesc:
        "Empêcher cette machine de se mettre en veille pendant les exécutions longues ou nocturnes. L'écran peut toujours s'obscurcir.",
      disableF12Title: 'Désactiver les outils de développement avec F12',
      disableF12Desc:
        "Empêcher F12 d'ouvrir les outils de développement. Ctrl+Maj+I (ou Cmd+Option+I sur Mac) continue de fonctionner.",
      attachmentSizeTitle: 'Taille maximale de chargement des aperçus / images',
      attachmentSizeDesc:
        "Taille maximale d'un fichier local que Desktop chargera pour les aperçus et les pièces jointes image, en Mo. La valeur par défaut est 16. Les pièces jointes distantes non-image utilisent une limite distincte de 256 Mo. Une valeur très élevée charge le fichier entier en mémoire et peut figer ou planter l'application.",
      attachmentSizeUnit: 'Mo',
      attachmentSizeLabel: 'Taille maximale de chargement des aperçus / images en mégaoctets',
      showOptions: 'Afficher les options'
    },
    hudModifier: {
      title: 'Toucher pour afficher le HUD',
      description:
        'Appuyez puis relâchez ⌘ + Option sur Mac, ou Ctrl + Alt sous Windows/Linux, pour afficher le HUD depuis n’importe quelle application. Désactivé par défaut ; s’applique uniquement à cet appareil.',
      permission:
        'Autorisez Hermes dans Réglages Système → Confidentialité et sécurité → Surveillance de l’entrée, puis réessayez. Ce geste n’enregistre pas les frappes et ne capture pas votre écran.',
      unavailable:
        'L’assistant du geste HUD n’a pas pu démarrer ou s’est arrêté de manière inattendue. Réessayez ou redémarrez Hermes. Le raccourci HUD existant fonctionne toujours dans Hermes.',
      missingHelper:
        'Il manque l’assistant du geste HUD dans cette installation de Hermes. Mettez à jour ou réinstallez Hermes, puis réessayez.',
      unsupportedSession:
        'Cette session de bureau ne prend pas en charge les appuis globaux sur les touches de modification. Linux nécessite X11 ; Wayland n’est pas pris en charge.'
    },
    screenshot: {
      enabledTitle: "Raccourci de capture d'écran",
      enabledDesc:
        "Appuyez simultanément sur les deux touches Commande depuis n'importe quelle application pour capturer sa fenêtre au premier plan et la joindre au brouillon Hermes actuel. Rien n'est envoyé automatiquement. Désactivé par défaut et limité à ce Mac. Le contenu peut être sensible : vérifiez la pièce jointe avant l'envoi.",
      statusTitle: "État du raccourci de capture d'écran",
      checking: "Vérification du raccourci de capture d'écran…",
      disabled: "Le raccourci de capture d'écran est désactivé.",
      starting: "Démarrage de l'écouteur du raccourci ; il n'est pas encore prêt.",
      ready: "Le raccourci est prêt. Les captures d'écran sont jointes au brouillon actuel sans être envoyées.",
      inputPermission:
        "L'autorisation Surveillance de l'entrée permet à Hermes de détecter les deux touches Commande lorsqu'une autre application est active. Autorisez Hermes dans Réglages Système → Confidentialité et sécurité → Surveillance de l'entrée, puis réessayez.",
      screenPermission:
        "L'autorisation Enregistrement de l'écran permet à Hermes de capturer la fenêtre au premier plan. Autorisez Hermes dans Réglages Système → Confidentialité et sécurité → Enregistrement de l'écran, puis réessayez. Redémarrez Hermes si macOS le demande.",
      openSettings: 'Ouvrir les Réglages Système',
      retry: 'Réessayer',
      unavailable: "Le raccourci de capture d'écran est indisponible. Réessayez ou désactivez-le.",
      errorTitle: "Erreur du raccourci de capture d'écran",
      loadFailed: "Impossible de lire l'état du raccourci. Réessayez pour vérifier son réglage actuel.",
      saveFailed: 'Impossible de confirmer la modification du raccourci. Réessayez pour vérifier son réglage actuel.',
      permissionFailed:
        "Impossible d'ouvrir les Réglages Système. Ouvrez manuellement Confidentialité et sécurité, puis réessayez.",
      captureFailed: "Impossible de capturer la fenêtre au premier plan. Rien n'a été joint ni envoyé.",
      contextChanged: "Le brouillon actuel a changé pendant la capture. L'image n'a pas été jointe ni envoyée."
    },
    quickEntry: {
      enabledTitle: 'Saisie rapide',
      enabledDesc:
        "Faites apparaître un petit compositeur depuis n'importe où avec un raccourci global et envoyez une invite sans ouvrir Hermes.",
      shortcutTitle: 'Raccourci de saisie rapide',
      shortcutDesc: 'Nécessite au moins un modificateur, par ex. CommandOrControl+Shift+Espace.',
      active: 'Le raccourci est actif.',
      takenBy: 'Une autre application utilise déjà ce raccourci — choisissez-en un autre.',
      invalidShortcut: 'Raccourci invalide. Incluez au moins une touche modificateur.'
    },
    credentials: {
      pasteKey: 'Coller la clé',
      pasteLabelKey: label => `Coller la clé ${label}`,
      optional: 'Facultatif',
      enterValueFirst: "Saisissez d'abord une valeur.",
      couldNotSave: "Impossible d'enregistrer l'identifiant.",
      remove: 'Supprimer',
      getKey: 'Obtenir une clé',
      saving: 'Enregistrement'
    },
    envActions: {
      actions: 'Actions',
      manageInKeys: 'Gérer dans les clés API',
      docs: 'Documentation',
      hideValue: 'Masquer la valeur',
      revealValue: 'Afficher la valeur',
      replace: 'Remplacer',
      set: 'Définir',
      clear: 'Effacer'
    },
    connections: {
      title: 'Connexions',
      intro:
        'Enregistrez tous les emplacements où vivent vos agents : cet appareil, les gateways distantes de votre réseau et les instances Hermes Cloud. Ils sont tous conservés ici.',
      stagedNote:
        "Les conversations et la liste des agents suivent la source choisie ; le backend de fenêtre géré par l'application reste sélectionné dans Paramètres → Gateway.",
      launchModeTitle: 'Au démarrage, revenir aux sessions de la dernière gateway utilisée',
      launchModeDesc: "Si cette option est désactivée, les sessions s'ouvrent sur la gateway principale.",
      searchPlaceholder: 'Rechercher des gateways…',
      noSearchResults: 'Aucune gateway ne correspond à votre recherche.',
      loadFailed: 'Impossible de charger les connexions',
      currentPill: 'Actuelle',
      primaryPill: 'Principale',
      managedPill: 'Cet appareil',
      addConnection: 'Ajouter une connexion',
      editConnection: 'Modifier',
      removeConnection: 'Supprimer',
      removeConfirmTitle: 'Supprimer cette connexion ?',
      removeConfirmDesc: label =>
        `« ${label} » sera supprimée de cette application. L'instance elle-même ne sera pas modifiée ; vous pourrez la rajouter à tout moment.`,
      makePrimary: 'Définir comme principale',
      testConnection: 'Tester',
      testOk: 'Accessible',
      testFailed: 'Le test de connexion a échoué',
      saveFailed: "Impossible d'enregistrer la connexion",
      removeFailed: 'Impossible de supprimer la connexion',
      updateAll: 'Mettre à jour toutes les instances',
      updateAllRunning: 'Mise à jour de toutes les instances…',
      updateAllDone: 'Mises à jour envoyées',
      updateAllFailed: "L'envoi groupé des mises à jour a échoué",
      updateSkippedCloud: 'Gérée par Hermes Cloud',
      kindLocal: 'Locale',
      kindRemote: 'Gateway distante',
      kindCloud: 'Hermes Cloud',
      kindSsh: 'SSH',
      kindLocalDesc: "L'environnement Hermes géré par cette application.",
      kindRemoteDesc: 'Une gateway Hermes accessible en HTTP(S), par le LAN, Tailscale ou Internet.',
      kindCloudDesc: 'Une instance hébergée découverte via votre compte Hermes Cloud.',
      kindSshDesc: 'Une installation Hermes accessible en SSH.',
      labelTitle: 'Nom',
      labelDesc:
        'Obligatoire. Affiché partout où cette instance apparaît et nécessairement unique (par exemple « Homelab » ou « PC professionnel »).',
      labelPlaceholder: 'Homelab',
      urlTitle: 'URL de la gateway',
      sshHostTitle: 'Hôte SSH',
      headersTitle: 'En-têtes supplémentaires du gateway',
      headersDesc:
        "Envoyés avec chaque requête HTTP et WebSocket vers ce gateway, notamment pour les proxys d'accès comme Cloudflare Access (CF-Access-Client-Id / CF-Access-Client-Secret). Les valeurs sont stockées chiffrées. Les en-têtes gérés par Hermes (Authorization, Cookie, Host…) sont ignorés.",
      headerValuePlaceholder: 'Valeur',
      headerValueSaved: 'Enregistrée — laissez vide pour la conserver',
      headerAdd: 'Ajouter un en-tête',
      headerRemove: 'Supprimer',
      duplicateLocal: "Cette application gère déjà une connexion locale — il ne peut y en avoir qu'une.",
      duplicateUrl: label => `Une connexion à cette URL de gateway existe déjà (« ${label} »).`,
      duplicateSsh: label => `Une connexion à cet hôte SSH existe déjà (« ${label} »).`,
      sameBackendHint: label => `Même backend que « ${label} »`,
      localAddHint:
        "La connexion locale est indisponible : la connexion gérée existe déjà (il ne peut y en avoir qu'une).",
      cloudAddHint:
        "Astuce : connectez-vous à Hermes Cloud ci-dessus pour découvrir automatiquement vos agents — utilisez ce formulaire uniquement pour enregistrer manuellement l'URL d'une instance connue.",
      save: 'Enregistrer la connexion',
      saving: 'Enregistrement…',
      cancel: 'Annuler',
      empty: 'Aucune connexion enregistrée.'
    },
    managedUpdates: {
      title: 'Mises à jour gérées',
      intro:
        'Mettez à jour de manière transactionnelle les installations SSH gérées par le Desktop : les sessions actives sont laissées se terminer, le dépôt distant est mis à jour, puis chaque profil est restauré avec un reçu corrélé.',
      sshConnection: 'Installation SSH gérée par le Desktop',
      update: 'Mettre à jour',
      updating: 'Mise à jour…',
      progress: 'Fin des sessions actives, mise à jour de l’installation distante et restauration des profils…',
      updated: 'Mise à jour terminée',
      partial: 'Mise à jour terminée — échec de la restauration',
      refused: 'Refusée',
      failed: 'Échec de la mise à jour',
      alreadyRunning: 'Une mise à jour est déjà en cours',
      receipt: (id: string, outcome: string) => `Reçu ${id} · ${outcome}`,
      receiptVersions: (pre: string, post: string) => `${pre} → ${post}`,
      scopesRestored: (profiles: string) => `Profils restaurés : ${profiles}`,
      scopeNotRestored: (profile: string, error: string) => `Profil « ${profile} » non restauré : ${error}`
    },
    gateway: {
      loading: 'Chargement des paramètres du gateway...',
      unavailableTitle: 'Paramètres du gateway indisponibles',
      unavailableDesc: "Le pont IPC du desktop n'expose pas les paramètres du gateway.",
      title: 'Connexion au gateway',
      envOverride: "remplacement par variable d'environnement",
      intro:
        'Local par défaut. Utilisez distant quand cette application doit piloter un backend Hermes ailleurs. Remplacements par profil ci-dessous.',
      envOverrideTitle: "Des variables d'environnement contrôlent cette session desktop.",
      envOverrideDesc:
        'Supprimez les variables HERMES_DESKTOP_REMOTE_URL et HERMES_DESKTOP_REMOTE_TOKEN pour utiliser le paramètre enregistré ci-dessous.',
      modeTitle: 'Mode de connexion',
      localTitle: 'Gateway local',
      localDesc:
        "Démarrer un backend Hermes privé sur localhost. C'est la valeur par défaut et cela fonctionne hors ligne.",
      remoteTitle: 'Gateway distant',
      remoteDesc: 'Connecter ce shell desktop à un backend Hermes distant.',
      remoteAuthHint:
        "Les gateways hébergés utilisent OAuth ou un nom d'utilisateur et un mot de passe ; les auto-hébergés peuvent utiliser un jeton de session.",
      cloudTitle: 'Hermes Cloud',
      cloudDesc:
        "Connectez-vous une fois à Hermes Cloud et choisissez parmi les agents de votre compte — pas d'URL à coller.",
      cloudSignInTitle: 'Hermes Cloud',
      cloudSignIn: 'Se connecter à Hermes Cloud',
      cloudSignedIn: 'Connecté à Hermes Cloud',
      cloudNeedsSignIn: 'Connectez-vous à Hermes Cloud pour découvrir les agents de votre compte.',
      cloudSignedInDesc:
        'Vous êtes connecté. Choisissez un agent ci-dessous ; la session se rafraîchit automatiquement.',
      cloudAgentsTitle: 'Vos agents',
      cloudOrgPickerTitle: 'Choisir une organisation',
      cloudOrgSelect: 'Sélectionner',
      cloudOrgChange: "Changer d'organisation",
      cloudOrgRole: role => `Rôle : ${role}`,
      cloudLoadingAgents: 'Chargement de vos agents…',
      cloudNoAgents: {
        before: 'Aucun agent trouvé sur ce compte… Créez-en un dans le ',
        linkText: 'portail Nous',
        after: ', puis actualisez.'
      },
      cloudRefresh: 'Actualiser',
      cloudConnect: 'Connecter',
      cloudSavedTitle: 'Gateways Cloud enregistrés',
      cloudSavedDesc:
        'Utilisez un gateway enregistré sans modifier celui par défaut. Connectez-vous ci-dessous pour ajouter des instances. Gérez leurs noms et leur connexion dans la liste des connexions enregistrées.',
      cloudUseSaved: 'Utiliser le gateway',
      cloudActive: 'Actif dans cette fenêtre',
      cloudConnecting: 'Connexion…',
      cloudDiscoverFailed: 'Impossible de charger vos agents Hermes Cloud',
      cloudConnectFailed: 'Impossible de se connecter à cet agent',
      cloudSignInFailed: 'Échec de la connexion à Hermes Cloud',
      cloudSignedOutTitle: 'Déconnecté de Hermes Cloud',
      cloudSignedOutMessage: 'Session Hermes Cloud effacée.',
      cloudConnectedTitle: 'Connecté',
      cloudConnectedPill: 'Connecté',
      cloudConnectedTo: name => `Connecté à ${name}.`,
      cloudAgentProvisioning: 'Provisionnement…',
      cloudStatusLabel: status => `État : ${status}`,
      remoteUrlTitle: 'URL distante',
      remoteUrlDesc:
        'URL de base du backend de tableau de bord distant. Les préfixes de chemin sont pris en charge, par exemple /hermes.',
      probing: "Vérification de la méthode d'authentification de ce gateway…",
      probeError:
        "Impossible d'atteindre ce gateway pour le moment. Vérifiez l'URL — la méthode d'authentification apparaîtra une fois qu'il répondra.",
      signedIn: 'Connecté',
      signIn: 'Se connecter',
      signOut: 'Se déconnecter',
      signInWith: provider => `Se connecter avec ${provider}`,
      authTitle: 'Authentification',
      authSignedInPassword:
        "Ce gateway utilise un nom d'utilisateur et un mot de passe. Vous êtes connecté ; la session se rafraîchit automatiquement.",
      authSignedInOauth: 'Ce gateway utilise OAuth. Vous êtes connecté ; la session se rafraîchit automatiquement.',
      authNeedsPassword:
        "Ce gateway utilise un nom d'utilisateur et un mot de passe. Connectez-vous pour autoriser cette application desktop.",
      authNeedsOauth: provider =>
        `Ce gateway utilise OAuth. Connectez-vous avec ${provider} pour autoriser cette application desktop.`,
      tokenTitle: 'Jeton de session',
      tokenDesc:
        "Le jeton de session du tableau de bord utilisé pour l'accès REST et WebSocket. Laissez vide pour conserver le jeton enregistré.",
      existingToken: value => `Jeton existant ${value}`,
      savedToken: 'enregistré',
      pasteSessionToken: 'Coller le jeton de session',
      plainTextConfirmTitle: 'Stocker le jeton du gateway en clair ?',
      plainTextConfirmDesc:
        'Aucun service de trousseau système n’a été trouvé sur cette machine : le jeton serait donc enregistré sans chiffrement dans le fichier des paramètres de connexion de l’application, lisible par tout processus exécuté par cet utilisateur. Installez ou activez le trousseau de votre système (GNOME Keyring ou KWallet sous Linux) pour un stockage chiffré.',
      plainTextConfirmAction: 'Enregistrer en clair',
      plainTextStoredTitle: 'Jeton stocké en clair',
      plainTextStoredDesc:
        'Le stockage sécurisé est indisponible : le jeton enregistré est donc stocké sans chiffrement dans le fichier des paramètres de connexion de l’application sur cette machine. Installez ou activez le trousseau de votre système (GNOME Keyring ou KWallet sous Linux) pour le chiffrer.',
      keychainEncryptionTitle: 'Chiffrer les secrets enregistrés avec le trousseau du système',
      keychainEncryptionDesc:
        "Désactivé par défaut. Une fois activé, les jetons du gateway et les identifiants de connexion sont chiffrés avec le trousseau du système (Trousseaux d'accès, GNOME Keyring ou Windows DPAPI) ; le système peut demander une autorisation ou un mot de passe. Sinon, ils sont stockés dans des fichiers ordinaires lisibles uniquement par votre compte utilisateur.",
      keychainEncryptionFailed: 'Impossible de modifier le chiffrement des secrets',
      testRemote: 'Tester le distant',
      saveForRestart: 'Enregistrer pour le prochain redémarrage',
      saveAndReconnect: 'Enregistrer et se reconnecter',
      diagnostics: 'Diagnostiques',
      diagnosticsDesc:
        'Afficher desktop.log dans votre gestionnaire de fichiers — utile quand le gateway ne démarre pas.',
      openLogs: 'Ouvrir les journaux',
      incompleteTitle: 'Gateway distant incomplet',
      incompleteSignIn: 'Saisissez une URL distante et connectez-vous avant de passer en distant.',
      incompleteToken: 'Saisissez une URL distante et un jeton de session avant de passer en distant.',
      incompleteSignInTest: 'Saisissez une URL distante et connectez-vous avant de tester.',
      incompleteTokenTest: 'Saisissez une URL distante et un jeton de session avant de tester.',
      enterUrlFirst: "Saisissez d'abord une URL distante.",
      restartingTitle: 'Reconnexion du gateway',
      savedTitle: 'Paramètres du gateway enregistrés',
      restartingMessage: 'Hermes Desktop va se reconnecter avec les paramètres enregistrés — le shell reste ouvert.',
      savedMessage: 'Enregistré pour le prochain redémarrage.',
      connectedTo: (baseUrl, version) => `Connecté à ${baseUrl}${version ? ` · Hermes ${version}` : ''}`,
      reachableTitle: 'Gateway distant accessible',
      signedOutTitle: 'Déconnecté',
      signedOutMessage: 'Session du gateway distant effacée.',
      failedLoad: 'Échec du chargement des paramètres du gateway',
      signInFailed: 'Échec de la connexion',
      signOutFailed: 'Échec de la déconnexion',
      testFailed: 'Échec du test du gateway distant',
      applyFailed: "Impossible d'appliquer les paramètres du gateway",
      saveFailed: "Impossible d'enregistrer les paramètres du gateway",
      sshTitle: 'Se connecter via SSH',
      sshDesc:
        "Hermes est lancé sur la machine distante via SSH et acheminé vers cette application — rien à démarrer ou exposer soi-même. Nécessite un accès SSH par clé fonctionnel vers l'hôte.",
      sshTrustHint:
        "La première clé d'hôte présentée est approuvée et fixée ; les changements ultérieurs échouent en mode fermé.",
      sshHostTitle: 'Hôte',
      sshHostDesc: 'utilisateur@hôte, ou un alias Host de ~/.ssh/config.',
      sshHostPick: 'Sélectionner un hôte…',
      sshHostPickTitle: 'Hôte',
      sshHostPickDesc: 'Un alias Host de ~/.ssh/config, ou Personnalisé pour en saisir un.',
      sshHostCustom: 'Personnalisé (saisie manuelle)…',
      sshUserTitle: 'Utilisateur',
      sshUserDesc: 'Vide = ~/.ssh/config ou votre utilisateur actuel.',
      sshUserPlaceholder: 'de ~/.ssh/config',
      sshPortTitle: 'Port',
      sshPortDesc: 'Vide = 22 ou le port de ~/.ssh/config.',
      sshKeyTitle: "Fichier d'identité",
      sshKeyDesc: 'Chemin de la clé privée. Vide = ssh-agent ou ~/.ssh/config.',
      sshHermesPathTitle: 'Chemin Hermes (facultatif)',
      sshHermesPathDesc: 'Chemin complet vers le binaire hermes distant. Vide = détection automatique.',
      sshHermesPathPlaceholder: 'détection automatique',
      sshTestConnection: 'Tester SSH',
      sshConnect: 'Se connecter',
      sshButtonsHint: "Enregistrer s'applique au prochain lancement. Connecter se reconnecte immédiatement.",
      sshReachable: (host, platform) => `Accessible : ${host} (${platform}) — Hermes trouvé`,
      sshIncompleteHost: 'Saisissez un hôte SSH avant de vous connecter.',
      sshErrUnreachable: "Impossible d'atteindre cet hôte via SSH. Vérifiez l'hôte, le port et votre réseau.",
      sshErrAuth:
        "Échec de l'authentification SSH. Chargez votre clé dans ssh-agent (ssh-add) ou définissez un IdentityFile dans ~/.ssh/config — Hermes exécute ssh de manière non interactive.",
      sshErrHostKey:
        "La clé de l'hôte a changé depuis votre dernière connexion. Vérifiez que ce changement est attendu, puis exécutez ssh-keygen -R <host> et reconnectez-vous.",
      sshErrNotInstalled:
        "Hermes n'est pas installé sur l'hôte distant. Installez-le là-bas (curl -fsSL https://hermes-agent.nousresearch.com/install.sh | sh) ou définissez le chemin Hermes.",
      sshErrPlatform:
        'Plateforme distante non prise en charge. Le mode SSH de Hermes Desktop supporte les hôtes distants Linux, macOS et Windows.',
      sshErrTimeout: "Expiration de la connexion SSH. L'hôte peut être inaccessible ou en veille.",
      sshErrUpdateRequired: "Mettez à jour Hermes sur l'hôte distant avant de vous connecter avec Desktop SSH.",
      sshErrUnknown: 'Échec de la connexion SSH.'
    },
    keys: {
      loading: 'Chargement des clés API et des identifiants...',
      failedLoad: 'Échec du chargement des clés API',
      empty: 'Rien configuré dans cette catégorie pour le moment.'
    },
    search: {
      placeholder: 'Rechercher dans tous les paramètres…',
      pill: 'Rechercher'
    },
    profileScope: {
      appliesTo: "S'applique à",
      editsProfile: profile => `Les modifications de cette page s'appliquent au profil « ${profile} ».`
    },
    mcp: {
      loading: 'Chargement des serveurs MCP...',
      invalidJson: 'MCP JSON invalide',
      saveFailed: "Échec de l'enregistrement",
      removeFailed: 'Échec de la suppression',
      reloadFailed: 'Échec du rechargement MCP',
      savedTitle: 'Serveur MCP enregistré',
      savedMessage: name => `${name} s'applique après le rechargement de MCP.`,
      disabled: 'désactivé',
      name: 'Nom',
      serverJson: 'JSON du serveur',
      remove: 'Supprimer',
      test: 'Tester la connexion',
      catalogLoading: 'Chargement du catalogue MCP...',
      catalogInstallFailed: name => `Échec de l'installation de ${name}`,
      catalogEnvRequired: "Remplissez les valeurs requises avant d'installer.",
      capabilitySummary: (tools, prompts, resources) =>
        `${[`${tools} outils`, ...(prompts ? [`${prompts} invites`] : []), ...(resources ? [`${resources} ressources`] : [])].join(', ')} activés`,
      costTokens: tokens => `~${tokens} jetons/appel`,
      usage30d: uses => `${uses} utilisations/30 j`,
      statusConnecting: 'Connexion…',
      statusNeedsAuth: 'Nécessite une authentification',
      statusError: 'Erreur',
      statusOff: 'Désactivé',
      allServers: 'Tous les serveurs',
      authenticatedTitle: 'Authentifié',
      authenticatedMessage: (server, count) => `${server} : ${count} outils`,
      authenticate: 'Authentifier',
      noOutput: 'Aucune sortie pour le moment.',
      deepLinkTitle: 'Ajouter un serveur MCP ?',
      deepLinkDescription:
        "Un lien demande d'ajouter ce serveur MCP à Hermes. Vérifiez attentivement la configuration ci-dessous : elle provient du lien, pas de Hermes.",
      deepLinkStdioWarning:
        'Ce serveur exécute sur votre machine un processus local avec la commande affichée ci-dessous. Continuez uniquement si vous faites confiance à sa source.',
      deepLinkConfirm: 'Ajouter le serveur',
      deepLinkNameInvalid: 'Les noms utilisent 1 à 64 lettres, chiffres, points, tirets ou underscores.',
      deepLinkNameConflict: name => `Un serveur nommé ${name} existe déjà — choisissez un autre nom ou annulez.`,
      deepLinkErrorTitle: "Lien d'installation MCP rejeté",
      deepLinkErrorName: 'Le nom du serveur est absent ou invalide dans le lien.',
      deepLinkErrorConfig: "La configuration du lien n'est pas un JSON valide encodé en base64.",
      deepLinkErrorShape: 'La configuration doit être un objet JSON comportant un champ chaîne `url` ou `command`.',
      deepLinkErrorUrl: 'Seules les URL de serveur http:// et https:// sont autorisées.',
      deepLinkErrorTooLarge: 'La configuration dépasse la limite de 32 Ko.'
    },
    model: {
      setupProviderFallback: 'fournisseur',
      setUpProvider: name => `Configurer ${name}`,
      staleAuxBefore: (count, names) =>
        count > 1
          ? `${count} tâches auxiliaires (${names}) s'exécutent encore sur `
          : `${count} tâche auxiliaire (${names}) s'exécute encore sur `,
      staleAuxAfter: ', et non sur votre modèle principal.',
      staleAuxOtherProviders: "d'autres fournisseurs",
      moaEnabled: 'Activé',
      moaSetDefault: 'Définir par défaut',
      moaNewPresetPlaceholder: 'nouveau préréglage',
      moaAddPreset: 'Ajouter un préréglage',
      customModel: 'Modèle personnalisé…',
      customModelPlaceholder: 'ID du modèle',
      chooseFromList: 'Choisir dans la liste',
      moaDefault: 'Par défaut :',
      moaReferenceToggle: (enabled, index) => `${enabled ? 'Désactiver' : 'Activer'} la référence ${index}`,
      moaReferenceTitle: index => `Référence ${index}`,
      moaAddReference: 'Ajouter un modèle de référence',
      loading: 'Chargement de la configuration du modèle...',
      appliesDesc:
        "S'applique aux nouvelles sessions. Utilisez le sélecteur de modèle dans le compositeur pour changer à chaud la conversation active.",
      provider: 'Fournisseur',
      model: 'Modèle',
      applying: 'Application...',
      defaultsLabel: 'Par défaut',
      reasoning: 'Raisonnement',
      reasoningOff: 'Désactivé',
      defaultsFailed: "Échec de l'enregistrement des modèles par défaut",
      loadFailed: 'Impossible de charger les modèles',
      restartRequired:
        "Ce backend exécute encore l'ancien code après une mise à jour. Redémarrez-le pour charger le nouveau code.",
      restartBackend: 'Redémarrer le backend',
      restartingBackend: 'Redémarrage du backend...',
      restartFailed: 'Impossible de redémarrer le backend',
      auxiliaryTitle: 'Modèles auxiliaires',
      resetAllToMain: 'Tout réinitialiser au principal',
      auxiliaryDesc:
        "Les tâches d'assistance s'exécutent sur le modèle principal par défaut. Attribuez un modèle dédié à toute tâche pour remplacer.",
      setToMain: 'Définir comme principal',
      change: 'Modifier',
      autoUseMain: 'auto · utiliser le modèle principal',
      inheritMainEffort: 'hériter · effort du modèle principal',
      providerDefault: '(par défaut du fournisseur)',
      fallbackAdd: 'Ajouter un secours',
      fallbackEmpty: "Aucun modèle de secours — le modèle par défaut est utilisé sauf en cas d'échec.",
      notInCatalog:
        "n'est pas dans la liste des modèles de ce fournisseur — les appels peuvent basculer sur un secours.",
      moaTitle: "Mélange d'agents",
      moaPreset: 'Préréglage',
      moaDescription:
        "Configurez des préréglages nommés qui apparaissent comme modèles chez le fournisseur Mélange d'agents. L'agrégateur est le modèle actif : il exécute chaque étape de la boucle d'outils et son fournisseur facture presque tout le coût de l'exécution. Par défaut, les modèles de référence ne donnent qu'un avis par tour utilisateur.",
      moaAggregator: 'Agrégateur',
      moaAggregatorBilled: "modèle actif · facturé pour l'exécution",
      moaReferenceHint: 'donne un avis une fois par tour par défaut',
      tasks: {
        vision: {
          label: 'Vision',
          hint: "Analyse d'image"
        },
        compression: {
          label: 'Compression',
          hint: 'Compaction de contexte'
        },
        skills_hub: {
          label: 'Hub de skills',
          hint: 'Recherche de skills'
        },
        approval: {
          label: 'Approbation',
          hint: 'Auto-approbation intelligente'
        },
        mcp: {
          label: 'MCP',
          hint: "Routage d'outils MCP"
        },
        title_generation: {
          label: 'Génération de titre',
          hint: 'Titres de session'
        },
        review: {
          label: 'Révision',
          hint: 'Sous-agent de révision /review'
        },
        triage_specifier: {
          label: 'Précision du triage',
          hint: 'Détail des spécifications Kanban'
        },
        kanban_decomposer: {
          label: 'Décomposition Kanban',
          hint: 'Décomposition des tâches'
        },
        profile_describer: {
          label: 'Description de profil',
          hint: 'Descriptions automatiques des profils'
        },
        curator: {
          label: 'Curateur',
          hint: "Revue d'utilisation des skills"
        }
      }
    },
    localModels: {
      connectionChanged: 'La connexion des modèles locaux a changé',
      title: 'Modèles locaux',
      runtimeTitle: 'Moteur local',
      runtimeReady: backend => `Prêt · ${backend}`,
      serverRunning: 'En cours',
      runtimeInstalled: 'Moteur llama.cpp installé',
      runtimeInstalledDetail: (tag, backend) =>
        `Build ${tag}, backend ${backend}. Hermes démarre et gère le serveur pour vous.`,
      installTitle: 'Installer le moteur local',
      installDetail:
        "Télécharge le moteur d'inférence llama.cpp (quelques centaines de Mo). Les modèles téléchargés s'exécutent entièrement sur cette machine : aucun compte requis et aucune donnée ne quitte votre ordinateur.",
      installAction: 'Installer le moteur',
      installing: 'Installation du moteur…',
      installFailed: "Échec de l'installation du moteur",
      hardwareTitle: 'Cette machine',
      hardwareLoading: 'Analyse de votre matériel…',
      vram: label => `${label} de mémoire GPU`,
      ram: label => `${label} de RAM`,
      unifiedMemory: 'Mémoire unifiée',
      modelsTitle: 'Modèles',
      recommended: 'Recommandé',
      recommendedReason: {
        'best-quality-resident':
          "Le modèle de meilleure qualité qui tient entièrement dans votre GPU et s'exécute à pleine vitesse. La sélection équilibre qualité et vitesse prévue sur ce matériel.",
        'speed-gated-quality':
          'Un modèle de meilleure qualité tient sur cette machine, mais sa bande passante mémoire le rendrait trop lent. Celui-ci est le meilleur modèle qui reste rapide.',
        'fastest-resident':
          "Aucun modèle n'atteint sa pleine vitesse sur ce matériel. Celui-ci s'en approche le plus tout en tenant entièrement dans la mémoire GPU."
      },
      noRecommendationTitle: 'Aucune recommandation automatique pour cette machine',
      noRecommendationDetail:
        "La configuration automatique nécessite un modèle présélectionné qui tient entièrement dans la mémoire GPU ou unifiée. Vous pouvez toujours choisir un modèle ci-dessous ou parcourir d'autres modèles.",
      noRecommendationAction: 'Parcourir les modèles',
      downloaded: 'Téléchargé',
      downloadAction: size => `Télécharger · ${size}`,
      downloadProgress: (done, total) => `Téléchargement de ${done} sur ${total}`,
      downloadDoneToast: model => `${model} est prêt.`,
      installDoneToast: 'Le moteur local est installé et prêt.',
      quickstartTitle: 'Exécuter un modèle sur cette machine',
      quickstartDetail: (model, size) =>
        `Un clic configure tout : le moteur local, ${model} (téléchargement de ${size}) et votre modèle par défaut pour les nouvelles conversations. Aucune donnée ne quitte cet ordinateur.`,
      quickstartDetailReady: model =>
        `Un clic définit ${model} comme modèle par défaut pour les nouvelles conversations. Tout s'exécute sur cette machine.`,
      quickstartAction: 'Configurer pour moi',
      quickstartConfigure: 'Configuration…',
      quickstartDoneToast: model =>
        `${model} est configuré : les nouvelles conversations s'exécutent sur cette machine.`,
      quickstartFailed: 'Échec de la configuration du modèle local',
      quickstartStageEngine: 'Moteur',
      quickstartStageModel: 'Modèle',
      quickstartStageFinish: 'Terminer',
      useAction: 'Utiliser',
      activePill: 'Par défaut',
      updateTitle: 'Mise à jour du moteur disponible',
      updateDetail: (next, current) =>
        `Une nouvelle build llama.cpp (${next}) est prête à être installée. Vous utilisez ${current}. Les modèles continuent de fonctionner pendant le téléchargement.`,
      updateAction: 'Mettre à jour le moteur',
      updating: 'Mise à jour du moteur…',
      upToDateTitle: 'Moteur à jour',
      upToDateDetail: (tag, backend) =>
        `llama.cpp ${tag} (${backend}) est en cours d'exécution : il s'agit de la dernière build fournie par Hermes.`,
      activeDetail:
        "Les nouvelles conversations utilisent ce modèle. Il se charge lors de l'envoi de votre premier message.",
      activeNotLoaded: 'Se charge avec votre premier message',
      loadedPill: 'En mémoire',
      placementResident: 'entièrement sur le GPU',
      placementSpilled: 'partiellement dans la RAM',
      placementResidentTip:
        'S’exécute entièrement dans la mémoire GPU avec cette fenêtre de contexte, à pleine vitesse.',
      placementSpilledTip:
        'Une partie de ce modèle s’exécute depuis la RAM système. Il fonctionne, mais plus lentement. Une build plus compacte ou un contexte plus petit tiendrait entièrement dans le GPU.',
      loadingPill: 'Chargement…',
      ejectTip: 'Libérer la mémoire GPU (le modèle se rechargera au prochain message)',
      ejected: 'Modèle déchargé : mémoire GPU libérée.',
      ejectFailed: 'Impossible de décharger le modèle',
      stopServer: 'Désactiver',
      startServer: 'Activer',
      runtimeRunningDetail:
        "Le serveur local est en cours d'exécution. Le désactiver libère toute la mémoire GPU et empêche les nouvelles conversations d'utiliser les modèles locaux jusqu'à sa réactivation.",
      serverStopped: 'Serveur local arrêté : mémoire GPU libérée.',
      serverStarted: 'Serveur local en cours d’exécution.',
      serverStopFailed: "Impossible d'arrêter le serveur local",
      serverStartFailed: 'Impossible de démarrer le serveur local',
      activating: 'Démarrage…',
      activateFailed: model => `Impossible de passer à ${model}`,
      activateDoneToast: model => `Les nouvelles conversations utilisent ${model}.`,
      downloadFailed: model => `Échec du téléchargement de ${model}`,
      pillFitsGpu: 'Tient dans votre GPU',
      pillUsesRam: 'Utilise la RAM système',
      pillTooBig: 'Trop volumineux pour cette machine',
      browseTitle: 'Trouver davantage de modèles',
      browseHint:
        'Recherchez dans tout Hugging Face. La taille des modèles téléchargés ici est automatiquement adaptée à votre machine, mais ils ne sont pas testés par notre équipe.',
      browsePlaceholder: 'Rechercher un modèle par nom ou auteur…',
      browseSearching: 'Recherche dans Hugging Face',
      browseListing: 'Lecture des fichiers du modèle',
      browseShowFiles: 'Afficher les fichiers',
      browseRefresh: 'Actualiser',
      browseDownloads: 'téléchargements',
      browseLikes: "mentions J'aime",
      browseGated: 'nécessite une connexion à Hugging Face',
      browseNoGguf: 'Aucun fichier de modèle compatible trouvé.',
      browseFitUnknown: 'Compatibilité inconnue',
      browseAlreadyDownloaded: 'Déjà téléchargé.',
      addedByYou: 'Ajouté par vous',
      browseDownloadStarted: 'Téléchargement de {name}',
      browseDownloadAria: 'Télécharger {name}',
      sideloadButton: 'Ajouter un fichier de modèle',
      sideloadTitle: 'Choisir un fichier de modèle GGUF',
      sideloadDone: '{name} ajouté.',
      sideloadAlreadyPresent: 'Déjà présent dans votre bibliothèque.',
      pillFullContext: max => `Contexte complet de ${max}`,
      pillFullContextTip: 'Utilise dès le départ la fenêtre de contexte complète du modèle',
      pillUpTo: max => `Contexte jusqu’à ${max}`,
      pillGrowsTip: 'Augmente automatiquement lorsque votre conversation a besoin de plus de place',
      pillVision: 'Comprend les images',
      deleteAction: 'Supprimer le modèle',
      deleteConfirm: model => `Supprimer ${model} du disque ?`,
      deleted: model => `${model} supprimé.`,
      deleteFailed: 'Échec de la suppression'
    },
    billing: {
      perMonth: (amount: string) => `${amount}/mois`,
      creditsPerMonth: (amount: string) => `${amount} crédits/mois`,
      usageLabel: (label: string) => `Utilisation ${label}`,
      freeTier: {
        signIn: 'Se connecter',
        title: 'Vous utilisez l’offre gratuite Nous',
        message: 'Connectez-vous avec un compte Nous pour débloquer davantage de modèles et d’outils.',
        caption:
          'Fonctionne avec nous/welcome, connecteurs inclus. La connexion conserve vos connecteurs et ajoute les outils qui nécessitent un compte ainsi que tous les autres modèles.',
        name: 'Nous · offre gratuite',
        footnote:
          'L’offre gratuite n’a ni solde ni rien à payer. Le paiement et l’utilisation apparaissent une fois connecté avec un compte Nous.',
        plan: 'Offre gratuite',
        model: 'Modèle',
        connectors: 'Connecteurs',
        included: 'Inclus'
      },
      amountValidation: {
        reloadTo: 'Recharger jusqu’à',
        greaterThanThreshold: 'Le montant de recharge doit être supérieur au seuil.',
        decimal: (label: string) => `${label} : saisissez un montant en dollars avec au plus 2 décimales.`,
        positive: (label: string) => `${label} : le montant doit être supérieur à 0 $.`,
        minimum: (label: string, amount: string) => `${label} : le minimum est de ${amount}.`,
        maximum: (label: string, amount: string) => `${label} : le maximum est de ${amount}.`
      },
      stepUp: {
        openVerification: 'Ouvrir la page de vérification',
        dismiss: 'Ignorer',
        waiting: 'En attente du lien de vérification…',
        verify: 'Vérifier pour continuer',
        deniedTitle: 'La vérification n’a pas été approuvée',
        deniedBody: 'La vérification s’est terminée sans autoriser les dépenses à distance pour ce terminal.',
        successTitle: 'Vérification terminée',
        successBody: 'Les dépenses à distance sont autorisées pour ce terminal.'
      },
      charge: {
        added: (amount?: string) => (amount ? `${amount} $ ajoutés.` : 'Crédits ajoutés.'),
        failedTitle: 'Échec du paiement',
        unconfirmedTitle: 'Résultat du paiement non confirmé',
        unconfirmedBody: (message: string) =>
          `${message} Le résultat de votre dernier paiement n’est pas confirmé - vérifiez votre solde/historique avant de réessayer.`,
        checkTitle: 'Impossible de vérifier le paiement',
        checkBody: 'Impossible de vérifier le paiement.',
        untrackedTitle: 'Le paiement n’a pas pu être suivi',
        untrackedBody: 'Le service de facturation a accepté la demande mais n’a renvoyé aucun identifiant de paiement.',
        timeoutTitle: 'Toujours en cours après 5 minutes',
        timeoutBody: 'Le paiement peut encore aboutir. Consultez le portail avant de réessayer.',
        authenticationRequired:
          'Votre banque exige une vérification (3DS). Terminez-la sur le portail pour finaliser cet achat.',
        expired: 'Votre carte a expiré. Mettez-la à jour sur le portail.',
        declined: 'Votre carte a été refusée. Essayez une autre carte sur le portail.',
        failedBody: (reason: string) => `Le paiement n’a pas abouti (${reason}).`
      },
      title: 'Facturation',
      preview: 'aperçu',
      summary: {
        balance: 'Solde',
        plan: 'Forfait',
        autoRefill: 'Recharge auto'
      },
      sections: {
        invoices: 'Factures',
        plan: 'Forfait',
        paymentAndCredits: 'Paiement et crédits',
        usage: 'Utilisation'
      },
      usage: {
        title: 'Utilisation'
      },
      buyCredits: {
        customAmount: 'Montant de crédits personnalisé',
        title: 'Acheter des crédits maintenant',
        buyButton: 'Acheter',
        processing: 'Traitement… vérification du règlement',
        added: (amount: string) => `${amount} ajoutés. Actualisation du solde.`,
        retry: 'Réessayer',
        openPortal: 'Ouvrir le portail'
      },
      plan: {
        title: 'Forfaits',
        changePlan: 'Changer de forfait',
        viewPlans: 'Voir les forfaits',
        backAria: 'Retour à la facturation',
        current: 'Forfait actuel',
        scheduled: 'Planifié',
        empty: 'Aucun forfait n’est disponible pour le moment.',
        undo: 'Annuler',
        undoing: 'Annulation…',
        downgrade: 'Rétrograder',
        confirmDowngrade: 'Confirmer la rétrogradation',
        tryAgain: 'Réessayer',
        checkingChange: 'Vérification de ce changement…',
        cannotChange: 'Ce changement ne peut pas être effectué ici.',
        alreadyOn: (name: string) => `Vous avez déjà le forfait ${name} — rien à changer.`,
        notScheduleable: 'Ce changement ne peut pas être planifié ici.',
        scheduling: 'Planification…',
        cancel: 'Annuler',
        effectScheduled: (targetName: string, effectiveAt: string, creditsDelta?: string) =>
          `Passage à ${targetName} — effectif ${effectiveAt}. Aucun débit maintenant ; vous conservez votre forfait actuel jusque-là.${creditsDelta ? ` Variation des crédits mensuels : ${creditsDelta}.` : ''}`
      },
      autoReload: {
        threshold: 'Seuil',
        thresholdAria: 'Seuil de recharge automatique',
        reloadTo: 'Recharger jusqu’à',
        reloadToAria: 'Montant cible de la recharge automatique',
        turnOffConfirm: 'Désactiver la recharge automatique ?',
        turnOff: 'Désactiver',
        disable: 'Désactiver',
        updated: 'Recharge automatique mise à jour.',
        turnedOff: 'Recharge automatique désactivée.',
        manage: 'Gérer',
        save: 'Enregistrer',
        saving: 'Enregistrement…',
        cancel: 'Annuler'
      },
      state: {
        notice: {
          loggedOut: {
            title: 'Connectez votre compte Nous',
            message:
              'Connectez-vous avec votre compte Nous pour voir ici votre solde, votre offre et votre utilisation.',
            action: 'Se connecter'
          },
          openPortal: 'Ouvrir le portail ↗',
          noCard: {
            title: 'Aucun moyen de paiement enregistré',
            message:
              'L’achat de crédits et la recharge automatique restent désactivés tant qu’aucune carte n’est enregistrée. Ajoutez-en une sur le portail.',
            action: 'Ajouter une carte ↗'
          }
        },
        paymentMethod: {
          title: 'Moyen de paiement',
          description: 'Gérez la carte utilisée pour les recharges et les renouvellements d’abonnement.',
          addAction: 'Ajouter un moyen de paiement',
          updateAction: 'Mettre à jour',
          provenance: {
            autoRefill: 'carte de recharge automatique',
            customerDefault: 'carte client par défaut',
            subPin: 'carte d’abonnement',
            suffix: (label: string) => ` - ${label}`
          }
        },
        buyCredits: {
          description: 'Un débit unique sur votre carte, ajouté à votre solde dès aujourd’hui.'
        },
        autoRefill: {
          title: 'Recharger quand le solde est bas',
          genericDescription: 'Maintenez votre solde approvisionné lorsqu’il passe sous votre seuil.',
          offPill: 'Désactivée',
          enabledPill: 'Activée',
          notAvailablePill: '—',
          manageCaption: 'Gérez la recharge automatique depuis le portail.',
          turnOnCaption: 'Activez la recharge automatique depuis le portail',
          chargesDescription: (reloadTo: string, threshold: string) =>
            `Débite automatiquement ${reloadTo} lorsque votre solde passe sous ${threshold}.`,
          distinctCardCaption: (cardLabel: string) =>
            `La recharge automatique débite ${cardLabel} — harmonisez sur le portail`,
          distinctCardFallback: 'une autre carte',
          reconcileAction: 'Harmoniser ↗'
        },
        usage: {
          subscriptionCredits: {
            title: 'Crédits d’abonnement',
            barLabel: 'Crédits d’abonnement restants',
            captionResets: (date: string) => `Réinitialisation ${date}`,
            valueOf: (remaining: string, monthly: string) => `${remaining} sur ${monthly} restants`,
            valueOver: (remaining: string, monthly: string, over: string) =>
              `${remaining} sur ${monthly} restants · ${over} de dépassement`
          },
          topupCredits: {
            title: 'Crédits rechargés',
            caption: 'N’expirent pas'
          },
          monthlyCap: {
            title: 'Plafond de dépenses mensuel',
            barLabel: 'Plafond de dépenses mensuel utilisé',
            captionDefault: 'Plafond par défaut',
            captionSpending: 'Dépenses mensuelles à distance',
            valueUsed: (spent: string, limit: string) => `${spent} sur ${limit} utilisés`
          }
        },
        planCard: {
          freeTier: 'Gratuit',
          chooseAction: 'Choisir ↗',
          adjustPlanAction: 'Ajuster le forfait ↗',
          unavailableCaption:
            'Les détails de l’abonnement sont indisponibles ; vous pouvez toujours ouvrir le portail.',
          downgradeCaption: (tierName: string, when: string) => `Passe à ${tierName} le ${when}.`,
          cancellationCaption: (when: string) => `Résiliation le ${when}.`,
          renewsCaption: (date: string) => `Renouvellement ${date}`,
          noSubscriptionCaption: 'Aucun abonnement actif — les modèles payants consomment les crédits rechargés.'
        }
      },
      errors: {
        consentRequired: {
          title: 'Confirmation de la carte requise',
          message: 'Confirmez cette carte pour les débits du terminal dans le portail'
        },
        insufficientScope: {
          title: 'Les dépenses à distance doivent être approuvées',
          message:
            'Cette action nécessite l’autorisation des dépenses à distance. Lancez une recharge pour l’autoriser, puis réessayez.'
        },
        remoteSpendingRevoked: {
          title: 'Les dépenses à distance ont été arrêtées',
          messageByAdmin: 'Un administrateur a arrêté les dépenses à distance pour ce terminal.',
          messageBySelf: 'Vous avez arrêté les dépenses à distance pour ce terminal.'
        },
        remoteSpendingReconnect: (who: string) =>
          `${who} Reconnectez-vous depuis Paramètres -> Gateway pour réautoriser cet appareil.`,
        sessionRevoked: {
          title: 'Session déconnectée',
          message: 'Votre session a été déconnectée. Reconnectez-vous depuis Paramètres → Gateway.'
        },
        cliBillingDisabled: {
          title: 'Les dépenses à distance sont désactivées',
          message:
            'Les dépenses à distance sont désactivées pour ce compte — un administrateur de facturation peut les activer depuis la page Hermes Agent du portail.'
        },
        roleRequired: {
          title: 'Rôle administrateur requis',
          message:
            'L’ajout de fonds nécessite un administrateur ou propriétaire de l’organisation. Demandez à un administrateur, ou gérez-le sur le portail.'
        },
        idempotencyConflict: {
          title: 'Lancez une nouvelle recharge',
          message: '🔴 Cette clé de paiement a déjà été utilisée pour un autre montant. Lancez une nouvelle recharge.'
        },
        noPaymentMethod: {
          title: 'Aucune carte enregistrée',
          message:
            '💳 Aucune carte enregistrée pour les débits du terminal. Configurez-en une sur le portail ' +
            '(les achats ponctuels de crédits n’enregistrent pas de carte réutilisable).'
        },
        orgAccessDenied: {
          title: 'Accès à l’organisation refusé',
          message: 'Ce jeton n’est lié à aucune organisation que vous pouvez gérer'
        },
        monthlyCapExceeded: {
          title: 'Plafond de dépenses mensuel atteint',
          messageReached: '🔴 Plafond de dépenses mensuel atteint.',
          messageHeadroom: (remaining: string) =>
            `🔴 Plafond de dépenses mensuel atteint — marge restante : ${remaining} $.`
        },
        rateLimited: {
          title: 'Trop de paiements pour le moment',
          message: (mins: number) =>
            mins > 0
              ? `🟡 Trop de paiements pour le moment (réessayez dans ~${mins} min). Il ne s’agit pas d’un échec de paiement.`
              : '🟡 Trop de paiements pour le moment. Il ne s’agit pas d’un échec de paiement.'
        },
        stripeUnavailable: {
          title: 'Stripe rencontre des difficultés',
          message: (mins: number) =>
            mins > 0
              ? `Stripe rencontre des difficultés — réessayez dans ~${mins} min`
              : 'Stripe rencontre des difficultés — réessayez sous peu'
        },
        upgradeCapExceeded: {
          title: 'Limite quotidienne de changements de forfait atteinte',
          message: 'Limite quotidienne de changements de forfait atteinte — réessayez demain'
        },
        endpointUnavailable: {
          title: 'Point de terminaison de facturation indisponible',
          message:
            'Le point de terminaison de facturation a renvoyé une réponse non JSON (il n’est peut-être pas disponible sur ce déploiement).'
        },
        timeout: {
          title: 'Délai de la requête de facturation dépassé',
          message: 'Délai de la requête de facturation dépassé.'
        },
        transport: {
          title: 'Échec de la connexion de facturation',
          message: 'La requête de facturation a échoué avant d’atteindre le gateway.'
        },
        default: {
          title: 'Échec de la requête de facturation',
          message: 'La requête de facturation a échoué.'
        }
      }
    },
    providers: {
      connectAccount: 'Connecter un compte',
      haveApiKey: 'Vous avez une clé API ?',
      intro:
        "Connectez-vous avec un abonnement — pas de clé API à copier. Hermes lance la connexion navigateur pour vous, directement dans l'application.",
      connected: 'Connecté',
      collapse: 'Réduire',
      connectAnother: 'Connecter un autre fournisseur',
      otherProviders: 'Autres fournisseurs',
      disconnect: 'Déconnecter',
      disconnectInTerminal: 'Déconnecter (exécute la commande de suppression dans le terminal)',
      removeConfirm: provider => `Supprimer ${provider} ?`,
      removeExternalGeneric: provider => `${provider} est géré par sa propre CLI — supprimez-le là-bas.`,
      removeKeyManaged: provider => `${provider} est configuré depuis une clé API. Supprimez-le depuis les clés API.`,
      removeTerminalConfirm: (provider, command) =>
        `Déconnecter ${provider} ? Cela exécute « ${command} » dans le terminal pour effacer l'identifiant.`,
      removeTerminalRunning: provider => `Exécution de la déconnexion ${provider} dans le terminal…`,
      removedTitle: 'Compte supprimé',
      removedMessage: provider => `${provider} a été supprimé.`,
      failedRemove: provider => `Impossible de supprimer ${provider}`,
      noProviderKeys: 'Aucune clé API de fournisseur disponible.',
      searchKeys: 'Rechercher des fournisseurs…',
      noKeysMatch: 'Aucun fournisseur ne correspond à votre recherche.',
      localEndpoint: {
        title: 'Point de terminaison local / personnalisé',
        description:
          "Pointez Hermes vers n'importe quel point de terminaison compatible OpenAI (Zyphra, vLLM, llama.cpp, Ollama, etc)."
      },
      loading: 'Chargement des fournisseurs...'
    },
    sessions: {
      loading: 'Chargement des sessions archivées…',
      archivedTitle: 'Sessions archivées',
      archivedIntro:
        'Les conversations archivées sont masquées de la barre latérale mais conservent tous leurs messages. Alt/⌥+Maj/⇧-clic sur une conversation de la barre latérale pour l’archiver.',
      emptyArchivedTitle: "Rien d'archivé",
      emptyArchivedDesc: 'Archivez une conversation pour la masquer ici.',
      unarchive: 'Restaurer',
      deletePermanently: 'Supprimer définitivement',
      messages: count => `${count} ${count === 1 ? 'message' : 'messages'}`,
      restored: 'Restauré',
      deleteConfirm: title => `Supprimer définitivement « ${title} » ? Cela ne peut pas être annulé.`,
      autoArchiveTitle: 'Auto-archiver les conversations inactives',
      autoArchiveDesc:
        "Archivez automatiquement les conversations que vous n'avez pas touchées depuis un moment. Les conversations épinglées ne sont jamais archivées et rien n'est supprimé — les conversations archivées se déplacent simplement ici.",
      autoArchiveDaysLabel: 'Archiver après',
      autoArchiveDaysUnit: "jours d'inactivité",
      autoArchiveFailed: "Impossible de mettre à jour l'auto-archivage",
      defaultDirTitle: 'Répertoire de projet par défaut',
      defaultDirDesc:
        'Les nouvelles sessions commencent dans ce dossier sauf si vous en choisissez un autre. Laissez-le non défini pour utiliser votre répertoire personnel.',
      defaultDirUpdated:
        "Répertoire de projet par défaut mis à jour — commencez une nouvelle conversation (Ctrl/⌘+N) pour qu'il prenne effet",
      defaultsTo: label => `Par défaut : ${label}.`,
      change: 'Modifier',
      choose: 'Choisir',
      clear: 'Effacer',
      notSet: 'Non défini',
      failedLoad: 'Impossible de charger les sessions archivées',
      unarchiveFailed: 'Échec de la restauration',
      deleteFailed: 'Échec de la suppression',
      updateDirFailed: 'Impossible de mettre à jour le répertoire par défaut',
      clearDirFailed: 'Impossible de vider le répertoire par défaut'
    },
    toolsets: {
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
  },
  skills: {
    tabSkills: 'Skills',
    tabToolsets: 'Outils',
    configuringProfile: 'Configuration de :',
    all: 'Tout',
    searchSkills: 'Rechercher des skills...',
    searchToolsets: 'Rechercher des outils...',
    refresh: 'Actualiser les skills',
    refreshing: 'Actualisation des skills',
    loading: 'Chargement des capacités...',
    noSkillsTitle: 'Aucun skill trouvé',
    noSkillsDesc: 'Essayez une recherche plus large ou une catégorie différente.',
    noToolsetsTitle: "Aucun ensemble d'outils trouvé",
    noToolsetsDesc: 'Essayez une requête de recherche plus large.',
    noDescription: 'Aucune description.',
    configured: 'Configuré',
    needsKeys: 'Nécessite des clés',
    visionModelHint:
      "La vision utilise votre configuration de modèle auxiliaire — le modèle capable d'images est choisi là-bas, pas par fournisseur ici.",
    visionModelLink: 'Choisissez le modèle de vision dans Paramètres → Modèles',
    toolsetsEnabled: (enabled, total) => `${enabled}/${total} ensembles d'outils activés`,
    configureToolset: label => `Configurer ${label}`,
    toggleToolset: (label, enabled) => `${enabled ? 'Activer' : 'Désactiver'} l'ensemble d'outils ${label}`,
    skillsLoadFailed: 'Échec du chargement des skills',
    toolsetsRefreshFailed: "Échec de l'actualisation des ensembles d'outils",
    skillEnabled: 'Skill activé',
    skillDisabled: 'Skill désactivé',
    toolsetEnabled: "Ensemble d'outils activé",
    toolsetDisabled: "Ensemble d'outils désactivé",
    appliesToNewSessions: name => `${name} s'applique aux nouvelles sessions.`,
    failedToUpdate: name => `Échec de la mise à jour de ${name}`,
    sortMostUsed: 'Plus utilisés',
    sortAlpha: 'A–Z',
    sortMostUsedDesc: '↓ Plus utilisés',
    sortLeastUsedAsc: '↑ Moins utilisés',
    enableAll: 'Tout activer',
    disableAll: 'Tout désactiver',
    disableUnused: 'Désactiver les inutilisés',
    bulkUpdated: count => `${count} ${count === 1 ? 'élément' : 'éléments'} mis à jour pour les nouvelles sessions.`,
    bulkNoChange: 'Rien à changer.',
    usageCount: count => `utilisé ${count}×`,
    provenance: {
      agent: 'Appris',
      bundled: 'Intégrés',
      hub: 'Hub'
    },
    emptyNoneFound: noun => `Aucun ${noun} trouvé`,
    emptyNothingMatches: query => `Rien ne correspond à « ${query} ».`,
    emptyNoneAvailable: noun => `Aucun ${noun} disponible pour le moment.`,
    changesApplyNewSessions: "Les modifications s'appliquent aux nouvelles sessions.",
    skillUpdated: 'Skill mis à jour',
    edit: 'Modifier',
    archive: 'Archiver',
    skillArchivedTitle: 'Skill archivé',
    skillArchivedMessage: 'Restaurable via hermes curator restore.',
    tabPlugins: 'Plugins',
    plugins: {
      agentTitle: "Plugins de l'agent",
      agentBlurb:
        "Étendez l'agent du profil sélectionné avec des outils, hooks et fournisseurs. Les changements prennent effet après le redémarrage du gateway.",
      pageBlurb:
        'Un plugin peut étendre cette application, l’agent ou les deux — chaque moitié a son propre interrupteur.',
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
      legacyBackend: 'Ce backend est trop ancien pour gérer les plugins depuis cet écran ; mettez Hermes à jour.',
      portableBadge: 'portable',
      serverStates: {
        connected: 'connecté',
        app_not_running: 'application non lancée',
        endpoint_unavailable: 'point de terminaison indisponible',
        no_interactive_session: 'aucune session interactive',
        version_too_old: 'version trop ancienne',
        missing_app: 'application manquante',
        unknown: 'état inconnu'
      },
      catalogTitle: 'Catalogue de plugins',
      catalogBrowse: 'Parcourir',
      catalogHide: 'Masquer le catalogue',
      catalogHint:
        'Utilisez « + Ajouter à cet agent » sur un plugin : les entrées vérifiées sont installées depuis leur commit épinglé dans le profil sélectionné.',
      alreadyInstalled: (name: string) => `${name} est déjà installé dans ce profil.`,
      catalogProvenance: (sha: string) =>
        `Installé depuis le catalogue Hermes${sha ? ` au commit épinglé ${sha}` : ''}.`,
      pinnedProvenance: (sha: string) =>
        `Épinglé au commit ${sha}. Les mises à jour sont refusées tant qu’il n’est pas réinstallé avec un nouvel épinglage.`,
      pinnedBadge: (sha: string) => `épinglé @ ${sha}`,
      tierOfficial: 'officiel',
      tierCommunity: 'communauté',
      updateToPin: (sha: string) => `Mettre à jour vers ${sha}`,
      updateFailed: (name: string) => `Impossible de mettre à jour ${name}`,
      updated: (name: string) =>
        `${name} mis à jour vers l’épinglage actuel du catalogue. Redémarrez le gateway pour l’appliquer.`,
      updateConsentTitle: (name: string) => `${name} demande davantage`,
      updateConsentBody: (name: string, sha: string) =>
        `Le nouvel épinglage de ${name} dans le catalogue (${sha}) ajoute des surfaces que la version installée n’a pas. Appliquez-le uniquement si vous leur faites confiance :`,
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
        optional: '(facultatif)',
        secretSet: '•••••••• (défini)',
        secretStoredAs: (env: string) =>
          `Stocké dans le .env du profil sous ${env}, jamais dans config.yaml ; laissez vide pour conserver la valeur actuelle.`
      }
    },
    officialCatalog: "Disponibles à l'installation",
    officialPill: 'Officiel',
    hub: {
      searchPlaceholder: 'Rechercher dans le hub de skills',
      search: 'Rechercher',
      searching: 'Recherche...',
      connectingHubs: 'Connexion aux hubs de skills...',
      connectedHubs: 'Hubs connectés :',
      featured: 'Skills en vedette',
      landingHint:
        "Recherchez dans le hub pour parcourir les skills installables depuis l'index officiel, GitHub et les sources communautaires.",
      noResults: 'Aucun skill correspondant trouvé dans le hub.',
      resultCount: (count, ms) => `${count} résultat${count === 1 ? '' : 's'}${ms !== null ? ` en ${ms} ms` : ''}`,
      timedOut: sources => `Expiration : ${sources}`,
      installed: 'Installé',
      install: 'Installer',
      installing: 'Installation...',
      uninstall: 'Désinstaller',
      uninstalling: 'Désinstallation...',
      updateAll: 'Mettre à jour les installés',
      updating: 'Mise à jour...',
      preview: 'Aperçu',
      scan: 'Analyser',
      scanning: 'Analyse...',
      close: 'Fermer',
      files: 'Fichiers',
      noReadme: "Ce skill n'a pas d'aperçu SKILL.md.",
      trust: {
        builtin: 'intégré',
        trusted: 'approuvé',
        community: 'communauté'
      },
      verdictSafe: 'Sûr',
      verdictCaution: 'Prudence',
      verdictDangerous: 'Dangereux',
      policyAllow: 'Installation autorisée',
      policyAsk: "Réviser avant d'installer",
      policyBlock: 'Installation bloquée par la politique',
      findings: count => `${count} résultat${count === 1 ? '' : 's'} de scan`,
      noFindings: 'Aucun résultat de scan de sécurité.',
      installStarted: name => `Installation de ${name}...`,
      uninstallStarted: name => `Désinstallation de ${name}...`,
      updateStarted: 'Mise à jour des skills installés...',
      actionFailed: "Échec de l'action du skill",
      installBlockedTitle: name => `Impossible d'installer ${name}`,
      installBlockedMessage: (findings, unverified) =>
        `Le scan de sécurité a signalé ${findings > 0 ? `${findings} élément${findings === 1 ? '' : 's'}` : 'des motifs risqués'} à examiner${unverified ? " et le skill provient d'une source non vérifiée" : ''}. Consultez le scan avant de décider si vous faites confiance à l'auteur.`,
      viewScan: 'Voir le scan',
      openLog: 'Ouvrir le journal',
      actionLog: "Journal d'action",
      alreadyInstalled: name => `« ${name} » est déjà installé`,
      pickerTitle: 'Hub de skills',
      pickerBrowse: 'Parcourir tout le hub',
      pickerHide: 'Masquer le navigateur du hub',
      pickerHint:
        'Cliquez sur « + Ajouter à cet agent » pour installer un skill : il apparaîtra ensuite dans la liste ci-dessus.',
      loadFailed: 'Échec du chargement du hub de skills',
      previewFailed: "Échec de l'aperçu du skill",
      scanFailed: 'Échec du scan de sécurité',
      searchFailed: 'Échec de la recherche dans le hub'
    }
  },
  starmap: {
    title: 'Graphique de mémoire',
    subtitle: (nodes, clusters) => `${nodes} skills dans ${clusters} catégories`,
    close: 'Fermer le graphique de mémoire',
    refresh: 'Actualiser',
    memory: 'Mémoire',
    filterAll: 'Tout',
    filterUsed: 'Utilisés',
    filterLearned: 'Appris',
    viewGraph: 'Graphique',
    loadFailed: 'Impossible de charger le graphique de mémoire',
    loading: 'Chargement…',
    emptyTitle: "Rien d'appris pour le moment",
    emptyDesc:
      'Au fur et à mesure que Hermes construit des skills et des mémoires pour votre travail, ils apparaissent ici.',
    share: 'Partager la carte',
    shareHint:
      'Copiez le code pour partager cette carte, ou collez-en un pour le charger. Il inclut uniquement la disposition, pas votre texte de mémoire ou de skill.',
    shareTitle: 'Importer / exporter la carte',
    sharePlaceholder: 'Collez un code de carte…',
    copy: 'Copier le code de la carte',
    copied: 'Copié !',
    importMap: 'Importer une carte',
    importBtn: 'Charger',
    importEmpty: 'Collez un code de carte pour le charger.',
    importSuccess: nodes => `Carte chargée avec ${nodes} ${nodes === 1 ? 'nœud' : 'nœuds'}.`,
    importedBadge: 'carte importée',
    resetToMine: 'Retour à ma carte'
  },
  agents: {
    extendedTranscript: 'Transcription étendue',
    transcriptTruncated: 'Affichage des 16 Kio les plus récents',
    transcriptUnavailable: 'Transcription en direct indisponible',
    close: 'Fermer les agents',
    title: 'Arbre de création',
    subtitle: 'Activité en direct des sous-agents pour le tour en cours.',
    emptyTitle: 'Aucun sous-agent en direct',
    emptyDesc: 'Quand un tour délègue du travail, les sous-agents diffusent leur progression ici.',
    running: 'En cours',
    failed: 'Échoué',
    done: 'Terminé',
    streaming: 'Diffusion',
    files: 'Fichiers',
    moreFiles: count => `+${count} fichiers supplémentaires`,
    moreAgents: count => `+${count} agents supplémentaires`,
    queued: 'En attente',
    waitingActivity: "En attente d'activité",
    steer: 'Orienter',
    steerPlaceholder: 'Instructions pour ce sous-agent',
    steerQueued: 'Instructions prévues au prochain point de contrôle',
    stopRequested: 'Arrêt demandé',
    requestRejected: "Le sous-agent n'a pas accepté la demande",
    delegation: index => `Délégation ${index}`,
    workers: count => `${count} travailleurs`,
    workersActive: count => `${count} actifs`,
    agentsCount: count => `${count} ${count === 1 ? 'agent' : 'agents'}`,
    activeCount: count => `${count} actifs`,
    failedCount: count => `${count} échoués`,
    toolsCount: count => `${count} outils`,
    filesCount: count => `${count} fichiers`,
    updatedAgo: age => `mis à jour il y a ${age}`,
    ageNow: 'maintenant',
    ageSeconds: seconds => `il y a ${seconds}s`,
    ageMinutes: minutes => `il y a ${minutes}m`,
    ageHours: hours => `il y a ${hours}h`,
    ageDays: days => `il y a ${days}j`,
    durationSeconds: seconds => `${seconds}s`,
    durationMinutes: (minutes, seconds) => `${minutes}m ${seconds}s`,
    tokens: value => `${value} tok`
  },
  commandCenter: {
    close: 'Fermer le centre de commandes',
    paletteTitle: 'Palette de commandes',
    back: 'Retour',
    searchPlaceholder: 'Rechercher des sessions, vues et actions',
    goTo: 'Aller à',
    goToSession: 'Aller à la session',
    branches: 'Branches',
    projects: 'Projets',
    openFolder: 'Ouvrir un dossier en tant que projet…',
    openFolderAt: path => `Ouvrir le dossier en tant que projet — ${path}`,
    newSessionInProject: project => `Nouvelle session dans ${project}`,
    commands: 'Commandes',
    startInBranch: branch => `Nouvelle conversation dans ${branch}`,
    commandCenter: 'Centre de commandes',
    appearance: 'Apparence',
    settings: 'Paramètres',
    changeTheme: 'Changer le thème',
    changeColorMode: 'Changer le mode couleur…',
    pets: {
      title: 'Animaux de compagnie',
      placeholder: 'Rechercher des animaux…',
      loading: 'Chargement de la galerie petdex…',
      error: "Impossible d'atteindre la galerie petdex…",
      staleBackend: 'Redémarrez Hermes pour utiliser les animaux — le backend précède cette fonctionnalité.',
      empty: 'Aucun animal correspondant.',
      turnOff: 'Éteindre',
      turnOn: 'Allumer',
      installed: 'Installé',
      generatedTag: 'Généré',
      adoptFailed: "Impossible d'adopter cet animal.",
      toggleFailed: enabled => `Impossible d'${enabled ? 'activer' : 'désactiver'} l'animal.`,
      noneAvailable: "Aucun animal disponible — choisissez-en un ci-dessous pour l'installer."
    },
    generatePet: {
      title: 'Générer un animal',
      placeholder: 'Décrivez un animal à générer…',
      promptHint: 'Saisissez une description, puis appuyez sur Entrée pour générer quatre apparences.',
      readyHint: 'Appuyez sur Entrée pour générer quatre apparences à partir de votre description.',
      generate: 'Générer',
      generating: 'Génération…',
      retry: 'Réessayer',
      hatch: 'Éclore',
      spawning: 'Création…',
      hatching: 'Éclosion de votre animal…',
      hatchingSub: 'Donner vie…',
      hatched: 'Il est éclos !',
      hatchRow: (_state, done, total) => `Dessin de l'image ${done} sur ${total}…`,
      hatchComposing: 'Assemblage…',
      hatchSaving: 'Presque terminé…',
      namePlaceholder: 'Nommez votre animal',
      staleBackend: 'Mettez à jour Hermes pour générer des animaux…',
      backgroundHint: "Vous pouvez fermer — Hermes vous notifiera quand c'est terminé.",
      slowProviderHint: 'Cela peut prendre plusieurs minutes',
      remix: 'Remixer',
      remixConfirmTitle: 'Remixer cette apparence ?',
      remixConfirmBody:
        'Cela génère un nouvel ensemble de projets en utilisant celui-ci comme point de départ. Cela peut prendre plusieurs minutes.',
      genericError: 'Échec de la génération — réessayez ou choisissez une suggestion.',
      referenceImageTooLarge: "L'image de référence est trop volumineuse. Utilisez-en une inférieure à 16 Mo.",
      referenceImageInvalid: 'Impossible de lire cette image de référence. Essayez un PNG, JPG, WebP ou GIF.',
      adopt: 'Adopter',
      startOver: 'Recommencer'
    },
    installTheme: {
      title: 'Installer un thème…',
      pageTitle: 'Installer un thème',
      placeholder: 'Rechercher dans le Marketplace VS Code…',
      loading: 'Recherche dans le Marketplace...',
      error: "Impossible d'atteindre le Marketplace.",
      empty: 'Aucun thème correspondant.',
      install: 'Installer',
      installing: 'Installation...',
      installed: 'Installé',
      installs: count => `${count} installations`
    },
    settingsFields: 'Champs de paramètres',
    mcpServers: 'Serveurs MCP',
    archivedChats: 'Conversations archivées',
    sections: {
      maintenance: 'Maintenance',
      sessions: 'Sessions',
      system: 'Système',
      usage: 'Utilisation'
    },
    sectionDescriptions: {
      maintenance: 'Diagnostiques, sauvegardes, curateur et données de mémoire',
      sessions: 'Rechercher et gérer les sessions',
      system: 'État, journaux et actions système',
      usage: 'Activité des jetons, coûts et skills au fil du temps'
    },
    nav: {
      newChat: {
        title: 'Nouvelle session',
        detail: 'Commencer une session fraîche'
      },
      settings: {
        title: 'Paramètres',
        detail: 'Configurer Hermes Desktop'
      },
      capabilities: {
        title: 'Capacités',
        detail: 'Skills, outils, serveurs MCP et plugins'
      },
      messaging: {
        title: 'Messagerie',
        detail: 'Configurer Telegram, Slack, Discord et plus'
      },
      artifacts: {
        title: 'Artéfacts',
        detail: 'Parcourir les sorties générées'
      }
    },
    sectionEntries: {
      sessions: {
        title: 'Panneau de sessions',
        detail: 'Rechercher, épingler et gérer les sessions'
      },
      system: {
        title: 'Panneau système',
        detail: 'État du gateway, journaux, redémarrage/mise à jour'
      },
      usage: {
        title: "Panneau d'utilisation",
        detail: 'Activité des jetons, coûts et skills'
      }
    },
    providerNavigate: 'Naviguer',
    providerSessions: 'Sessions',
    refresh: 'Actualiser',
    refreshing: 'Actualisation...',
    noResults: 'Aucun résultat correspondant trouvé.',
    pinSession: 'Épingler la session',
    unpinSession: 'Désépingler la session',
    exportSession: 'Exporter la session',
    deleteSession: 'Supprimer la session',
    noSessions: 'Aucune session pour le moment.',
    gatewayRunning: 'Gateway de messagerie en cours',
    gatewayStopped: 'Gateway de messagerie arrêté',
    hermesActiveSessions: (version, count) => `Hermes ${version} · Sessions actives ${count}`,
    restartGateway: 'Redémarrer le gateway',
    openBrowser: 'Ouvrir le navigateur',
    gatewayRestartFailed: 'Échec du redémarrage du gateway.',
    sharedGatewayRestartTitle: 'Redémarrer le gateway partagé ?',
    sharedGatewayRestartDescription: bots => `Tous les bots de cet appareil se reconnecteront : ${bots}`,
    sharedGatewayRestartConfirm: 'Tout redémarrer',
    sharedGatewayRestarted: count => `Gateway partagé redémarré (${count} ${count === 1 ? 'bot' : 'bots'})`,
    updateHermes: 'Mettre à jour Hermes',
    reloadWindow: 'Recharger la fenêtre',
    actionRunning: 'en cours',
    actionDone: 'terminé',
    actionFailed: 'échoué',
    actionStartedWaiting: 'Action démarrée, attente du statut...',
    loadingStatus: 'Chargement du statut...',
    recentLogs: 'Journaux récents',
    noLogs: 'Aucun journal chargé pour le moment.',
    days: count => `${count}j`,
    statSessions: 'Sessions',
    statApiCalls: 'Appels API',
    statTokens: 'Jetons entrants/sortants',
    statCost: 'Coût estim.',
    actualCost: cost => `réel ${cost}`,
    loadingUsage: "Chargement de l'utilisation...",
    noUsage: period => `Aucune utilisation au cours des ${period} derniers jours.`,
    retry: 'Réessayer',
    dailyTokens: 'Jetons quotidiens',
    input: 'entrant',
    output: 'sortant',
    noDailyActivity: 'Aucune activité quotidienne.',
    topModels: 'Principaux modèles',
    noModelUsage: 'Aucune utilisation de modèle pour le moment.',
    topSkills: 'Principaux skills',
    noSkillActivity: 'Aucune activité de skill pour le moment.',
    actions: count => `${count} actions`,
    logFile: 'Fichier journal',
    logLevel: 'Niveau',
    logSearchPlaceholder: 'Filtrer les lignes du journal...',
    maintenance: {
      runOps: 'Diagnostiques',
      doctor: 'Exécuter le diagnostic',
      doctorDesc: "Vérifier l'installation, la configuration et les fournisseurs",
      securityAudit: 'Audit de sécurité',
      securityAuditDesc: 'Analyser la configuration et les skills à la recherche de paramètres risqués',
      backup: 'Créer une sauvegarde',
      backupDesc: 'Archiver en zip la configuration, les mémoires, les skills et les sessions',
      debugShare: 'Partage de débogage',
      debugShareDesc:
        'Téléverser un rapport expurgé + journaux, obtenir des liens partageables (auto-suppression après 6h)',
      debugShareRunning: 'Téléversement du rapport de débogage...',
      debugShareLinks: 'Liens de partage',
      debugShareFailed: 'Échec du partage de débogage',
      copyLink: 'Copier le lien',
      linkCopied: 'Lien copié',
      curator: 'Curateur de skills',
      curatorDesc: 'Revue en arrière-plan qui archive les skills agent obsolètes',
      curatorPaused: 'En pause',
      curatorActive: 'Actif',
      curatorDisabled: 'Désactivé',
      curatorLastRun: when => `Dernière exécution ${when}`,
      curatorNeverRan: 'Jamais exécuté',
      pause: 'Mettre en pause',
      resume: 'Reprendre',
      runNow: 'Exécuter maintenant',
      memoryData: 'Données de mémoire',
      memoryDataDesc: 'Fichiers de mémoire intégrés injectés dans chaque session',
      memoryProvider: name => `Fournisseur actif : ${name}`,
      builtinMemory: 'intégré',
      memoryFile: "Mémoire de l'agent (MEMORY.md)",
      userFile: 'Profil utilisateur (USER.md)',
      bytes: size => size,
      empty: 'vide',
      resetMemory: 'Réinitialiser la mémoire',
      resetUser: 'Réinitialiser le profil',
      resetAll: 'Réinitialiser les deux',
      resetConfirm: target => `Supprimer ${target} ? Cela ne peut pas être annulé.`,
      resetDone: files => `Supprimé ${files}.`,
      resetFailed: 'Échec de la réinitialisation de la mémoire',
      actionStarted: name => `${name} démarré — suivi du journal...`,
      actionFailed: name => `${name} a échoué au démarrage`,
      running: 'En cours...',
      viewLog: "Journal d'action"
    }
  },
  messaging: {
    search: 'Rechercher dans la messagerie...',
    loading: 'Chargement des plateformes de messagerie...',
    loadFailed: 'Échec du chargement des plateformes de messagerie',
    states: {
      connected: 'Connecté',
      connecting: 'Connexion',
      disabled: 'Désactivé',
      fatal: 'Erreur',
      gateway_stopped: 'Gateway de messagerie arrêté',
      not_configured: 'Nécessite une configuration',
      pending_restart: 'Redémarrage requis',
      retrying: 'Nouvelle tentative',
      startup_failed: 'Échec du démarrage'
    },
    unknown: 'Inconnu',
    hintPendingRestart: "Redémarrez le gateway depuis la barre d'état pour appliquer ce changement.",
    sharedListenerUrl: 'Servi sur le listener partagé du gateway à',
    hintGatewayStopped: "Démarrez le gateway depuis la barre d'état pour vous connecter.",
    credentialsSet: 'Identifiants définis',
    needsSetup: 'Nécessite une configuration',
    gatewayStopped: 'Gateway de messagerie arrêté',
    getCredentials: 'Obtenez vos identifiants',
    openSetupGuide: 'Ouvrir le guide de configuration',
    required: 'Requis',
    recommended: 'Recommandé',
    advanced: count => `Avancé (${count})`,
    noTokenNeeded:
      'Cette plateforme ne nécessite pas de jeton ici. Utilisez le guide de configuration ci-dessus, puis activez-la ci-dessous.',
    enabled: 'Activé',
    disabled: 'Désactivé',
    unsavedChanges: 'Modifications non enregistrées',
    saving: 'Enregistrement...',
    saveChanges: 'Enregistrer les modifications',
    saved: 'Enregistré',
    replaceValue: 'Remplacer la valeur actuelle',
    openDocs: 'Ouvrir la documentation',
    clearField: key => `Effacer ${key}`,
    enableAria: name => `Activer ${name}`,
    disableAria: name => `Désactiver ${name}`,
    platformEnabled: name => `${name} activé`,
    platformDisabled: name => `${name} désactivé`,
    restartToApply: 'Ce changement prend effet après un redémarrage du gateway.',
    setupSaved: name => `Configuration ${name} enregistrée`,
    restartToReconnect: 'Les nouveaux identifiants prennent effet après un redémarrage du gateway.',
    appliedLive: 'Appliqué au gateway en cours d’exécution.',
    connectingLive: 'Le gateway en cours d’exécution se connecte avec les nouveaux identifiants.',
    keyCleared: key => `${key} effacé`,
    setupUpdated: name => `La configuration ${name} a été mise à jour.`,
    failedUpdate: name => `Échec de la mise à jour de ${name}`,
    failedSave: name => `Échec de l'enregistrement de ${name}`,
    failedClear: key => `Échec de l'effacement de ${key}`,
    pendingRequests: count => `Demandes en attente (${count})`,
    pendingAria: count =>
      `${count} ${count === 1 ? "demande d'appairage en attente" : "demandes d'appairage en attente"}`,
    approvedUsers: count => `Utilisateurs approuvés (${count})`,
    approve: 'Approuver',
    approving: 'Approbation…',
    revoke: 'Révoquer',
    revoking: 'Révocation…',
    revokeAria: name => `Révoquer l'accès de ${name}`,
    revokeTitle: "Révoquer l'accès",
    revokeDesc: name => `${name} perdra l'accès et ne sera plus reconnu à partir de son prochain message.`,
    approvedUser: name => `${name} approuvé`,
    approvedHint: 'Cet utilisateur sera reconnu automatiquement à partir de son prochain message.',
    revokedUser: name => `Accès de ${name} révoqué`,
    failedApprove: name => `Échec de l'approbation de ${name}`,
    failedRevoke: name => `Échec de la révocation de l'accès de ${name}`,
    pairingLockedOut:
      "Trop d'échecs d'approbation — cette plateforme est temporairement verrouillée. Réessayez plus tard.",
    waitingSince: minutes => (minutes < 1 ? "à l'instant" : `il y a ${minutes} min`),
    restartNeeded: 'Enregistré. Redémarrez le gateway de messagerie pour appliquer les nouveaux paramètres.',
    restartNow: 'Redémarrer maintenant',
    restarting: 'Redémarrage…',
    restartFailedManual: 'Le redémarrage du gateway a échoué — redémarrez-le manuellement et consultez ses journaux.',
    restartFailedManualDetail:
      'Réessayez le redémarrage ; si le problème persiste, ouvrez les journaux et envoyez les diagnostics.',
    restartAgain: 'Redémarrer à nouveau',
    openLogs: 'Ouvrir les journaux',
    telegramQr: {
      title: 'Choisissez comment connecter votre bot Telegram',
      subtitle:
        "Les deux méthodes connectent un bot que vous contrôlez et enregistrent ses identifiants uniquement dans cette installation d'Hermes.",
      quickSetup: 'Configuration rapide',
      recommended: 'Recommandé',
      quickHelp:
        "Scannez un code QR et confirmez dans Telegram. Hermes crée le bot et détecte automatiquement votre identifiant d'utilisateur Telegram.",
      createWithQr: 'Créer avec un code QR',
      starting: 'Démarrage…',
      replaceWarning:
        "Des identifiants Telegram sont déjà configurés. Une nouvelle configuration par QR ou un nouveau jeton remplacera le bot actuel lors de l'enregistrement.",
      scanHint: "Scannez avec l'application Telegram de votre téléphone ou ouvrez le lien sur cet ordinateur.",
      waiting: 'En attente de Telegram…',
      expiresIn: remaining => `Expire dans ${remaining}`,
      expired: 'Expiré',
      openTelegram: 'Ouvrir Telegram',
      ready: 'Bot créé',
      allowedUsers: 'Utilisateurs autorisés',
      ownerDetected: 'Propriétaire détecté',
      addAtLeastOne: "Ajoutez au moins un identifiant d'utilisateur Telegram.",
      userIdPlaceholder: "Identifiant d'utilisateur Telegram",
      add: 'Ajouter',
      numericOnly: "Les identifiants d'utilisateurs Telegram autorisés doivent être numériques.",
      saveAndRestart: 'Enregistrer et redémarrer',
      applying: 'Enregistrement…',
      pairingExpired: 'La liaison Telegram a expiré. Relancez une configuration par QR pour réessayer.',
      stillWaiting: detail => `Toujours en attente de Telegram. Nouvelle tentative après : ${detail}`,
      savedRestarting: 'Telegram enregistré ; redémarrage du gateway…',
      savedRestartFailed: detail => `Telegram enregistré ; le redémarrage du gateway a échoué${detail}`
    },
    fieldCopy: {
      TELEGRAM_BOT_TOKEN: {
        label: 'Jeton du bot',
        help: "Créez un bot avec @BotFather, puis collez le jeton qu'il vous fournit.",
        placeholder: 'Coller le jeton du bot Telegram'
      },
      TELEGRAM_ALLOWED_USERS: {
        label: "IDs d'utilisateurs Telegram autorisés",
        help: "Recommandé. IDs numériques séparés par des virgules depuis @userinfobot. Sans cela, n'importe qui peut envoyer un message privé à votre bot."
      },
      TELEGRAM_PROXY: {
        label: 'URL du proxy',
        help: 'Uniquement nécessaire sur les réseaux où Telegram est bloqué.'
      },
      DISCORD_BOT_TOKEN: {
        label: 'Jeton du bot',
        help: 'Créez une application dans le Discord Developer Portal, ajoutez un bot, puis collez son jeton.'
      },
      DISCORD_ALLOWED_USERS: {
        label: "IDs d'utilisateurs Discord autorisés",
        help: "Recommandé. IDs d'utilisateurs Discord séparés par des virgules."
      },
      DISCORD_REPLY_TO_MODE: {
        label: 'Style de réponse',
        help: 'first, all ou off.'
      },
      DISCORD_ALLOW_ALL_USERS: {
        label: 'Autoriser tous les utilisateurs Discord',
        help: "Développement uniquement. Quand vrai, n'importe qui peut envoyer un message privé au bot sans liste d'autorisation."
      },
      DISCORD_HOME_CHANNEL: {
        label: "ID du canal d'accueil",
        help: 'Canal où le bot envoie des messages proactifs (sortie cron, rappels).'
      },
      DISCORD_HOME_CHANNEL_NAME: {
        label: "Nom du canal d'accueil",
        help: "Nom d'affichage pour le canal d'accueil dans les journaux et la sortie d'état."
      },
      BLUEBUBBLES_ALLOW_ALL_USERS: {
        label: 'Autoriser tous les utilisateurs iMessage',
        help: "Quand vrai, ignorez la liste d'autorisation BlueBubbles."
      },
      MATTERMOST_ALLOW_ALL_USERS: {
        label: 'Autoriser tous les utilisateurs Mattermost'
      },
      MATTERMOST_HOME_CHANNEL: {
        label: "Canal d'accueil"
      },
      QQ_ALLOW_ALL_USERS: {
        label: 'Autoriser tous les utilisateurs QQ'
      },
      QQBOT_HOME_CHANNEL: {
        label: "Canal d'accueil QQ",
        help: 'Canal ou groupe par défaut pour la livraison cron.'
      },
      QQBOT_HOME_CHANNEL_NAME: {
        label: "Nom du canal d'accueil QQ"
      },
      SLACK_BOT_TOKEN: {
        label: 'Jeton du bot Slack',
        help: 'Utilisez le jeton du bot depuis OAuth & Permissions après avoir installé votre application Slack.',
        placeholder: 'Coller le jeton du bot Slack'
      },
      SLACK_APP_TOKEN: {
        label: "Jeton de l'application Slack",
        help: 'Utilisez le jeton de niveau application requis pour le Socket Mode.',
        placeholder: "Coller le jeton de l'application Slack"
      },
      SLACK_ALLOWED_USERS: {
        label: "IDs d'utilisateurs Slack autorisés",
        help: "Recommandé. IDs d'utilisateurs Slack séparés par des virgules."
      },
      MATTERMOST_URL: {
        label: 'URL du serveur',
        placeholder: 'https://mattermost.example.com'
      },
      MATTERMOST_TOKEN: {
        label: 'Jeton du bot'
      },
      MATTERMOST_ALLOWED_USERS: {
        label: "IDs d'utilisateurs autorisés",
        help: "Recommandé. IDs d'utilisateurs Mattermost séparés par des virgules."
      },
      MATRIX_HOMESERVER: {
        label: 'URL du homeserver',
        placeholder: 'https://matrix.org'
      },
      MATRIX_ACCESS_TOKEN: {
        label: "Jeton d'accès"
      },
      MATRIX_USER_ID: {
        label: "ID d'utilisateur du bot",
        placeholder: '@hermes:example.org'
      },
      MATRIX_ALLOWED_USERS: {
        label: "IDs d'utilisateurs Matrix autorisés",
        help: "Recommandé. IDs d'utilisateurs séparés par des virgules au format @utilisateur:serveur."
      },
      SIGNAL_HTTP_URL: {
        label: 'URL du pont Signal',
        placeholder: 'http://127.0.0.1:8080',
        help: "URL d'un pont REST signal-cli en cours d'exécution."
      },
      SIGNAL_ACCOUNT: {
        label: 'Numéro de téléphone',
        help: 'Le numéro enregistré avec votre pont signal-cli.'
      },
      SIGNAL_ALLOWED_USERS: {
        label: 'Utilisateurs Signal autorisés',
        help: 'Recommandé. Identifiants Signal séparés par des virgules.'
      },
      WHATSAPP_ENABLED: {
        label: 'Activer le pont WhatsApp',
        help: "Défini automatiquement par l'interrupteur ci-dessous. Ne touchez pas sauf si vous savez que vous en avez besoin."
      },
      WHATSAPP_MODE: {
        label: 'Mode du pont'
      },
      WHATSAPP_ALLOWED_USERS: {
        label: 'Utilisateurs WhatsApp autorisés',
        help: 'Recommandé. Numéros de téléphone ou IDs WhatsApp séparés par des virgules.'
      }
    },
    platformIntro: {}
  },
  webhooks: {
    search: 'Rechercher des webhooks...',
    loading: 'Chargement des webhooks...',
    loadFailed: 'Échec du chargement des webhooks',
    subscriptions: (count: number) => `Abonnements (${count})`,
    hint: "Les modifications d'abonnement rechargent à chaud une fois que le récepteur est en cours d'exécution. Les abonnements désactivés rejettent les événements entrants.",
    empty: 'Aucun abonnement webhook pour le moment.',
    disabledTitle: 'Récepteur de webhooks désactivé',
    disabledBody:
      "Les webhooks sont leur propre plateforme gateway. Activez-les ici pour accepter les événements HTTP entrants ; les canaux de conversation ne sont nécessaires que lorsqu'un abonnement livre vers Telegram, Discord, Slack ou un autre canal.",
    enable: 'Activer les webhooks',
    enabling: 'Activation...',
    enabled: (name: string) => `Activé : « ${name} »`,
    disabled: (name: string) => `Désactivé : « ${name} »`,
    enableRow: 'Activer',
    disableRow: 'Désactiver',
    delete: 'Supprimer',
    deleting: 'Suppression...',
    deleted: 'Webhook supprimé',
    deleteTitle: 'Supprimer le webhook',
    deleteDescPrefix: 'Cela supprimera définitivement ',
    deleteDescSuffix: '. Cela ne peut pas être annulé.',
    deleteFailed: (name: string) => `Échec de la suppression de « ${name} »`,
    toggleFailed: (name, enabled) => `Échec : ${enabled ? 'activation' : 'désactivation'} de « ${name} »`,
    newSubscription: 'Nouvel abonnement',
    restarting: 'Redémarrage du gateway...',
    restartNeeded:
      "Les webhooks sont activés, mais le gateway a toujours besoin d'un redémarrage avant que le récepteur puisse se connecter.",
    restartGateway: 'Redémarrer le gateway',
    restartingGateway: 'Redémarrage...',
    restartFailed: (detail: string) => `Échec du redémarrage du gateway${detail}`,
    enabledRestarting: 'Webhooks activés ; redémarrage du gateway...',
    all: '(tous)',
    deliverOnly: 'livraison uniquement',
    createdTitle: 'Abonnement créé',
    createdSecretHint: "Copiez le secret maintenant — il n'est affiché qu'une fois.",
    webhookUrl: 'URL du webhook',
    secretOnce: 'Secret (affiché une fois)',
    done: 'Terminé',
    fieldName: 'Nom',
    fieldNamePlaceholder: 'ex. github-push',
    fieldDescription: 'Description',
    fieldDescriptionPlaceholder: 'Ce que fait ce webhook (facultatif)',
    fieldEvents: 'Événements',
    fieldEventsPlaceholder: 'séparés par des virgules, laissez vide pour tous',
    fieldSkills: 'Skills',
    fieldSkillsPlaceholder: 'noms de skills séparés par des virgules (facultatif)',
    fieldDeliver: 'Livrer vers',
    fieldDeliverOnly: 'Livrer le payload uniquement',
    fieldPrompt: 'Invite',
    fieldPromptPlaceholder: "Instructions pour l'agent quand ce webhook se déclenche (facultatif)",
    nameRequired: 'Nom requis',
    create: 'Créer',
    creating: 'Création...',
    created: 'Créé',
    createFailed: (detail: string) => `Échec de la création : ${detail}`,
    copy: 'Copier',
    deliverOptions: {
      log: 'Journal',
      telegram: 'Telegram',
      discord: 'Discord',
      slack: 'Slack',
      email: 'E-mail',
      github_comment: 'Commentaire GitHub'
    }
  },
  profiles: {
    close: 'Fermer les profils',
    nameHint: 'Lettres minuscules, chiffres, tirets et underscores. Doit commencer par une lettre ou un chiffre.',
    title: 'Profils',
    count: count => `${count} ${count === 1 ? 'profil' : 'profils'}`,
    search: 'Rechercher des profils...',
    loading: 'Chargement des profils...',
    newProfile: 'Nouveau profil',
    importProfile: 'Importer un profil…',
    exportProfile: 'Exporter le profil…',
    imported: 'Profil importé',
    exported: 'Profil exporté',
    failedImport: "Échec de l'importation du profil",
    failedExport: "Échec de l'exportation du profil",
    allProfiles: 'Tous les profils',
    showAllProfiles: 'Afficher tous les profils',
    switchToProfile: name => `Basculer vers ${name}`,
    switchToConnection: name => `Basculer vers ${name}`,
    switchConnectionFailed: name => `Impossible de se connecter à ${name}`,
    manageProfiles: 'Gérer les profils…',
    connectGateway: 'Connecter un autre gateway Hermes…',
    fleet: {
      allOnGateway: 'Tous les profils de cette gateway',
      gateway: gateway => `Profils sur ${gateway}`,
      gatewayUnreachable: gateway => `${gateway} · inaccessible`,
      onGateway: (name, gateway) => `${name} · ${gateway}`,
      switchTo: (name, gateway) => `Basculer vers ${name} sur ${gateway}`,
      deleteOn: gateway => ` sur ${gateway}`,
      localDevice: 'Cet appareil (backend local — installe Hermes s’il manque, sinon ouvre une nouvelle session)',
      switchDeviceTitle: 'Basculer vers cet appareil ?',
      switchDeviceDesc:
        'Cela ouvre une nouvelle session sur cet ordinateur. La conversation en cours reste sur l’autre gateway.',
      switchDeviceConfirm: 'Basculer',
      installDeviceTitle: 'Basculer vers cet appareil ?',
      installDeviceDesc:
        'Hermes sera installé localement, puis une nouvelle session s’ouvrira sur cet ordinateur. Rien n’est installé tant que vous ne confirmez pas.',
      installDeviceConfirm: 'Installer localement',
      connectExistingInstead: 'Connecter un existant à la place'
    },
    status: {
      unread: (count: number) => (count === 1 ? '1 session non lue' : `${count} sessions non lues`),
      needsInput: (count: number) =>
        count === 1 ? '1 session attend votre réponse' : `${count} sessions attendent votre réponse`,
      working: (count: number) => (count === 1 ? '1 session en cours' : `${count} sessions en cours`)
    },
    remoteOverride: {
      menuItem: 'Se connecter à un hôte distant…',
      badge: (host: string) => `S'exécute sur ${host}`,
      title: (profile: string) => `Connecter ${profile} à un hôte distant`,
      description:
        "Les sessions de ce profil s'exécuteront sur le système Hermes distant indiqué, plutôt que sur cet ordinateur.",
      urlLabel: 'Adresse distante',
      urlPlaceholder: 'https://hermes.exemple.fr',
      urlInvalid: 'Saisissez une adresse complète commençant par http:// ou https://',
      tokenLabel: "Jeton d'accès",
      tokenPlaceholder: 'Collez le jeton de session distant',
      tokenSavedHint: 'Un jeton est déjà enregistré. Laissez ce champ vide pour le conserver.',
      plainTextOptIn:
        "Cet ordinateur ne dispose d'aucun stockage sécurisé pour les clés. Le jeton serait donc enregistré sans chiffrement sur le disque. L'enregistrer quand même.",
      collisionWarning: (label: string) =>
        `Un gateway nommé « ${label} » existe déjà dans les paramètres. Cette connexion de profil est indépendante et ne le modifiera pas.`,
      confirmTitle: 'Connecter ce profil à un hôte distant ?',
      confirmNote: (profile: string, host: string) =>
        `Les nouvelles conversations de ${profile} s'exécuteront sur ${host}. Cet ordinateur distant exécutera les commandes et lira les fichiers qui s'y trouvent, pas ceux de cet appareil. Connectez-vous uniquement à un hôte de confiance.`,
      confirmBack: 'Retour',
      connect: 'Connecter',
      connecting: 'Connexion…',
      disconnect: 'Supprimer la connexion distante',
      savedTitle: 'Profil connecté',
      savedMessage: (profile: string, host: string) => `${profile} s'exécute désormais sur ${host}`,
      removedTitle: 'Connexion distante supprimée',
      removedMessage: (profile: string) => `${profile} s'exécute désormais sur cet ordinateur`,
      removeFailed: 'Impossible de supprimer la connexion distante',
      authFailedTitle: "L'hôte distant a refusé le jeton enregistré",
      authFailedMessage: (profile: string, host: string) =>
        `${host} a refusé le jeton enregistré pour ${profile}. Il a peut-être été modifié sur le système distant.`,
      updateToken: 'Saisir un nouveau jeton…'
    },
    actions: 'Actions',
    color: 'Couleur…',
    colorFor: 'Couleur',
    openInNewWindow: 'Ouvrir dans une nouvelle fenêtre',
    setAsDefault: 'Définir par défaut',
    defaultProfile: 'Profil par défaut',
    defaultSet: name => `${name} est maintenant le profil par défaut`,
    defaultDescription:
      "Utilisé à l'ouverture de Hermes et pour les nouvelles conversations. Les sessions existantes restent dans leur profil.",
    failedSetDefault: 'Impossible de définir le profil par défaut',
    setColor: color => `Définir la couleur ${color}`,
    autoColor: 'Auto',
    noProfiles: 'Aucun profil pour le moment.',
    selectPrompt: 'Sélectionnez un profil pour afficher ses détails.',
    refresh: 'Actualiser les profils',
    refreshing: 'Actualisation des profils',
    default: 'par défaut',
    skills: count => `${count} ${count === 1 ? 'skill' : 'skills'}`,
    env: 'env',
    defaultBadge: 'Par défaut',
    rename: 'Renommer',
    renameMenu: 'Renommer…',
    exportMenu: 'Exporter',
    editSoul: 'Modifier SOUL.md…',
    copySetup: 'Copier la configuration',
    copying: 'Copie en cours…',
    modelLabel: 'Modèle',
    skillsLabel: 'Skills',
    notSet: 'Non défini',
    soulDesc: "L'invite système et les instructions de persona intégrées dans ce profil.",
    soulOptional: 'facultatif',
    soulPlaceholder: mode =>
      `L'invite système / persona pour ce profil.\nLaissez vide pour conserver la valeur par défaut ${mode}.`,
    soulPlaceholderCloned: 'cloné',
    soulPlaceholderEmpty: 'vide',
    unsavedChanges: 'Modifications non enregistrées',
    loadingSoul: 'Chargement de SOUL.md...',
    emptySoul: 'SOUL.md vide — commencez à écrire le persona...',
    saving: 'Enregistrement...',
    saveSoul: 'Enregistrer SOUL.md',
    deleteTitle: 'Supprimer le profil ?',
    deleteDescPrefix: 'Cela supprimera ',
    deleteDescMid: ' et supprimera son ',
    deleteDescSuffix: ' répertoire. Cela ne peut pas être annulé.',
    deleting: 'Suppression...',
    createDesc: 'Les profils sont des environnements Hermes indépendants : configuration, skills et SOUL.md séparés.',
    nameLabel: 'Nom',
    cloneFrom: 'Cloner depuis',
    cloneFromNone: 'Aucun (vide)',
    cloneFromDesc: 'Copie la configuration, les skills et le SOUL.md depuis le profil source sélectionné.',
    cloneFromDefault: 'Cloner depuis le profil par défaut',
    cloneFromDefaultDesc: 'Copier la configuration, les skills et le SOUL.md depuis votre profil par défaut.',
    invalidName: hint => `Nom invalide. ${hint}`,
    nameRequired: 'Le nom est requis.',
    creating: 'Création...',
    createAction: 'Créer le profil',
    renameTitle: 'Renommer le profil',
    renameDescPrefix: "Le renommage met à jour le répertoire du profil et les scripts d'enveloppe dans ",
    renameDescSuffix: '.',
    displayNameTitle: 'Nommer cet agent',
    displayNameDesc:
      "Définit le nom affiché dans toute l'application. L'identifiant interne du profil reste « default ».",
    displayNameLabel: "Nom d'affichage",
    newNameLabel: 'Nouveau nom',
    renaming: 'Renommage...',
    created: 'Profil créé',
    renamed: 'Profil renommé',
    deleted: 'Profil supprimé',
    setupCopied: 'Commande de configuration copiée',
    soulSaved: 'SOUL.md enregistré',
    failedLoad: 'Échec du chargement des profils',
    failedDelete: 'Échec de la suppression du profil',
    failedCopy: 'Échec de la copie de la commande de configuration',
    failedLoadSoul: 'Échec du chargement de SOUL.md',
    failedSaveSoul: "Échec de l'enregistrement de SOUL.md",
    failedCreate: 'Échec de la création du profil',
    failedRename: 'Échec du renommage du profil'
  },
  modelAssignment: {
    saveFailed: 'Hermes n’a pas enregistré ce changement de modèle.',
    confirmTitle: 'Avertissement sur le choix du modèle',
    confirmDetail: 'Confirmez uniquement si vous acceptez ce compromis.',
    confirmAction: 'Confirmer',
    declined:
      'Changement de modèle annulé — vous avez refusé l’avertissement sur le niveau d’entraînement sur les données.'
  },
  cron: {
    close: 'Fermer le cron',
    title: 'Tâches planifiées',
    count: count => `${count} ${count === 1 ? 'tâche' : 'tâches'}`,
    search: 'Rechercher des tâches cron...',
    loading: 'Chargement des tâches cron...',
    states: {
      enabled: 'activée',
      scheduled: 'planifiée',
      running: 'en cours',
      paused: 'en pause',
      disabled: 'désactivée',
      error: 'erreur',
      completed: 'terminée'
    },
    lastRunFailed: 'Échec de la dernière exécution :',
    editJob: 'Modifier la tâche',
    runAgain: 'Relancer',
    deliveryLabels: {
      local: 'Ce bureau',
      telegram: 'Telegram',
      discord: 'Discord',
      slack: 'Slack',
      email: 'E-mail'
    },
    scheduleLabels: {
      daily: 'Quotidien',
      weekdays: 'Jours ouvrés',
      weekly: 'Hebdomadaire',
      monthly: 'Mensuel',
      hourly: 'Toutes les heures',
      'every-15-minutes': 'Toutes les 15 minutes',
      custom: 'Personnalisé'
    },
    scheduleHints: {
      daily: 'Chaque jour à 9h00',
      weekdays: 'Du lundi au vendredi à 9h00',
      weekly: 'Chaque lundi à 9h00',
      monthly: 'Le premier jour de chaque mois à 9h00',
      hourly: "En début d'heure",
      'every-15-minutes': 'Toutes les 15 minutes',
      custom: 'Syntaxe cron ou langage naturel'
    },
    days: {
      '0': 'Dimanche',
      '1': 'Lundi',
      '2': 'Mardi',
      '3': 'Mercredi',
      '4': 'Jeudi',
      '5': 'Vendredi',
      '6': 'Samedi',
      '7': 'Dimanche'
    },
    dayFallback: value => `jour ${value}`,
    everyDayAt: time => `Chaque jour à ${time}`,
    weekdaysAt: time => `Jours ouvrés à ${time}`,
    everyDayOfWeekAt: (day, time) => `Chaque ${day} à ${time}`,
    monthlyOnDayAt: (dayOfMonth, time) => `Mensuellement le jour ${dayOfMonth} à ${time}`,
    topOfHour: "En début d'heure",
    everyHourAt: minute => `Toutes les heures à :${minute}`,
    newCron: 'Nouveau cron',
    emptyDescNew:
      "Planifiez une invite à exécuter selon une expression cron. Hermes l'exécutera et livrera les résultats vers la destination de votre choix.",
    emptyDescSearch: 'Essayez une requête de recherche plus large.',
    emptyTitleNew: 'Aucune tâche planifiée pour le moment',
    emptyTitleSearch: 'Aucune correspondance',
    last: 'Dernière :',
    next: 'Prochaine :',
    overdueSince: 'En retard depuis :',
    noRuns: 'Aucune exécution',
    manage: 'Gérer',
    showRuns: 'Afficher les exécutions',
    hideRuns: 'Masquer les exécutions',
    runHistory: 'Historique des exécutions',
    actionsTitle: 'Actions de la tâche cron',
    resume: 'Reprendre le cron',
    pause: 'Mettre en pause le cron',
    resumeTitle: 'Reprendre',
    pauseTitle: 'Mettre en pause',
    triggerNow: 'Déclencher maintenant',
    edit: 'Modifier le cron',
    deleteTitle: 'Supprimer la tâche cron ?',
    deleteDescPrefix: 'Cela supprimera ',
    deleteDescSuffix: ' définitivement. Elle cessera de se déclencher immédiatement.',
    deleting: 'Suppression...',
    resumed: 'Cron repris',
    paused: 'Cron en pause',
    triggered: 'Cron déclenché',
    deleted: 'Cron supprimé',
    created: 'Cron créé',
    updated: 'Cron mis à jour',
    failedLoad: 'Échec du chargement des tâches cron',
    failedUpdate: 'Échec de la mise à jour de la tâche cron',
    failedTrigger: 'Échec du déclenchement de la tâche cron',
    failedDelete: 'Échec de la suppression de la tâche cron',
    failedSave: "Échec de l'enregistrement de la tâche cron",
    editTitle: 'Modifier la tâche cron',
    createTitle: 'Nouvelle tâche cron',
    editDesc:
      "Mettre à jour le planning, l'invite ou la cible de livraison. Les modifications s'appliquent à la prochaine exécution.",
    createDesc:
      'Planifiez une invite à exécuter automatiquement. Utilisez la syntaxe cron ou une phrase naturelle comme « toutes les 15 minutes ». ',
    nameLabel: 'Nom',
    namePlaceholder: 'Point du matin',
    promptLabel: 'Invite',
    promptPlaceholder: 'Résumez mes fils Slack non lus et envoyez-moi les 5 principaux par email...',
    frequencyLabel: 'Fréquence',
    deliverLabel: 'Livrer vers',
    deliverNeedsHomeChannel: "définissez d'abord un canal d'accueil",
    modelLabel: 'Modèle',
    modelDefault: 'Par défaut (modèle global)',
    customScheduleLabel: 'Planning personnalisé',
    customPlaceholder: '0 9 * * * ou jours ouvrés à 9h',
    customHint: 'Expression cron ou phrases comme « toutes les heures » ou « jours ouvrés à 9h ». ',
    optional: 'Facultatif',
    promptRequired: "L'invite est requise.",
    promptScheduleRequired: "L'invite et le planning sont requis.",
    scheduleRequired: 'Le planning est requis.',
    scriptOnlyEditHint: "Tâche script uniquement (pas d'invite IA). ID de tâche :",
    saveChanges: 'Enregistrer les modifications',
    createAction: 'Créer le cron',
    tabs: {
      jobs: 'Tâches',
      blueprints: 'Plans'
    },
    blueprints: {
      tab: 'Plans',
      startFrom: 'Commencer à partir de',
      custom: 'Personnalisé',
      subtitle: "Automatismes prêts à l'emploi",
      dialogDesc: 'Remplissez les détails et planifiez-les.',
      scheduleIt: 'Planifiez-le',
      scheduling: 'Planification...',
      scheduled: 'Plan planifié',
      loading: 'Chargement des plans...',
      failedLoad: 'Échec du chargement des plans',
      emptyTitle: 'Aucun plan disponible',
      emptyDesc: "Aucun plan d'automatisation n'est disponible sur ce backend."
    }
  },
  artifacts: {
    search: 'Rechercher des artefacts...',
    refresh: 'Actualiser les artefacts',
    refreshing: 'Actualisation des artefacts',
    indexing: 'Indexation des artefacts de session récents',
    tabAll: 'Tout',
    tabImages: 'Images',
    tabFiles: 'Fichiers',
    tabLinks: 'Liens',
    noArtifactsTitle: 'Aucun artefact trouvé',
    noArtifactsDesc:
      'Les images générées et les sorties de fichier apparaîtront ici au fur et à mesure que les sessions les produisent.',
    failedLoad: 'Échec du chargement des artefacts',
    openFailed: "Échec de l'ouverture",
    itemsImage: 'images',
    itemsLink: 'liens',
    itemsFile: 'fichiers',
    itemsGeneric: 'éléments',
    zero: '0',
    rangeOf: (start, end, total) => `${start}-${end} sur ${total}`,
    goToPage: (itemLabel, page) => `Aller à ${itemLabel}, page ${page}`,
    colTitleLink: 'Titre du lien',
    colTitleFile: 'Nom',
    colTitleDefault: 'Titre / nom',
    colLocationLink: 'URL',
    colLocationFile: 'Chemin',
    colLocationDefault: 'Emplacement',
    colSession: 'Session',
    kindImage: 'image',
    kindFile: 'fichier',
    kindLink: 'lien',
    chat: 'Conversation',
    copyUrl: "Copier l'URL",
    copyPath: 'Copier le chemin'
  },
  artifactCard: {
    kind: {
      code: 'Code',
      html: 'Page interactive',
      svg: 'Graphique'
    },
    generating: lines => `Génération… ${lines} lignes`,
    versionBadge: count => `${count} versions`,
    open: 'Ouvrir'
  },
  artifactPreview: {
    versionOf: (current, total) => `v${current} sur ${total}`,
    olderVersion: 'Version précédente',
    newerVersion: 'Version suivante',
    latest: 'Dernière',
    copyContent: 'Copier le contenu',
    download: 'Télécharger',
    openInBrowser: 'Ouvrir dans le navigateur',
    openInBrowserFailed: "Impossible d'ouvrir dans le navigateur",
    missingTitle: 'Artefact indisponible',
    missingBody: 'Cet artefact ne figure plus dans le registre local.'
  },
  sidebar: {
    filter: {
      grouping: 'Regroupement',
      ordering: 'Tri',
      show: 'Afficher',
      filters: 'Filtres',
      status: 'État',
      pullRequest: 'Pull request',
      profile: 'Profil',
      project: 'Projet',
      archived: 'Archivées',
      resetToDefaults: 'Rétablir les valeurs par défaut',
      expandAll: 'Tout déplier',
      collapseAll: 'Tout replier',
      inboxStyle: 'Style boîte de réception',
      updated: 'Mise à jour',
      created: 'Création',
      tokens: 'Jetons',
      cost: 'Coût',
      manual: 'Manuel',
      preview: 'Aperçu',
      pr: 'PR',
      needsInput: 'Action requise',
      working: 'En cours',
      unread: 'Non lues',
      draft: 'Brouillon',
      idle: 'Inactive',
      open: 'Ouverte',
      merged: 'Fusionnée',
      closed: 'Fermée',
      noPR: 'Aucune PR'
    },
    gatewayGroups: {
      grouping: 'Gateway et profil',
      rename: 'Renommer le groupe',
      aliasLabel: "Nom d'affichage",
      aliasHint: "Seul le nom d'affichage change ; les noms du gateway et du profil restent inchangés.",
      resetName: 'Rétablir le nom',
      moveUp: 'Déplacer vers le haut',
      moveDown: 'Déplacer vers le bas',
      reorder: 'Réordonner le groupe',
      actions: 'Actions du groupe'
    },
    profileRail: 'Barre des profils',
    nav: {
      'new-session': 'Nouvelle session',
      capabilities: 'Capacités',
      messaging: 'Messagerie',
      artifacts: 'Artefacts',
      cron: 'Tâches planifiées'
    },
    searchAria: 'Rechercher des sessions',
    searchPlaceholder: 'Rechercher des sessions…',
    clearSearch: 'Effacer la recherche',
    noMatch: query => `Aucune session ne correspond à « ${query} ».`,
    results: 'Résultats',
    pinned: 'Épinglées',
    sessions: 'Sessions',
    terminal: 'Terminal',
    files: 'Fichiers',
    review: 'Revue',
    logs: 'Journaux',
    cronJobs: 'Tâches cron',
    groupAriaGrouped: 'Afficher les sessions en une seule liste',
    groupAriaUngrouped: 'Grouper les sessions par espace de travail',
    showProjects: 'Afficher les projets',
    showSessions: 'Afficher les sessions',
    groupTitleGrouped: 'Dégrouper les sessions',
    groupTitleUngrouped: 'Grouper par espace de travail',
    allPinned: "Tout est épinglé ici. Désépinglez une conversation pour l'afficher dans les récentes.",
    shiftClickHint: "Shift-clic sur une conversation pour l'épingler",
    noWorkspace: 'Aucun espace de travail',
    projectEmpty: 'Aucune session pour le moment',
    projectLoadFailed: 'Impossible de charger les sessions',
    noSessions: 'Aucune session pour le moment',
    storageCorrupt: {
      title: 'La base de données des sessions est endommagée',
      body: (profiles: string) =>
        `Hermes ne peut pas lire tout l’historique des sessions de ${profiles}. Les conversations absentes de cette liste n’ont pas été supprimées ; le fichier qui les contient est endommagé.`,
      action: 'Quittez Hermes sur ce profil, puis inspectez le fichier sans le modifier, ou restaurez un instantané :',
      guide: 'Guide de récupération'
    },
    noFilterMatches: 'Aucune session ne correspond à ces filtres',
    projects: {
      showAllSessions: 'Afficher toutes les sessions',
      sectionLabel: 'Projets',
      home: 'Accueil',
      autoDiscovered: 'Détecté automatiquement',
      newButton: 'Nouveau projet',
      createTitle: 'Nouveau projet',
      createDesc: 'Nommez un espace de travail et ajoutez un ou plusieurs dossiers.',
      renameTitle: 'Renommer le projet',
      addFolderTitle: 'Ajouter un dossier',
      namePlaceholder: 'ex. Projet secret',
      foldersLabel: 'Dossiers',
      ideaLabel: 'Idée',
      ideaPlaceholder: "De quoi s'agit-il ? (enregistré dans IDEA.md)",
      ideaGenerate: 'Générer une idée',
      ideaGenerating: 'Génération…',
      ideaShuffle: 'Mélanger les modèles',
      noFolders: 'Aucun dossier ajouté pour le moment.',
      addFolder: 'Ajouter un dossier',
      primaryBadge: 'principal',
      removeFolder: 'Supprimer',
      create: 'Créer',
      menu: 'Actions',
      menuRename: 'Renommer',
      menuAppearance: 'Apparence',
      noColor: 'Aucune couleur',
      menuAddFolder: 'Ajouter un dossier',
      menuSetActive: 'Définir comme actif',
      menuDelete: 'Supprimer',
      moveToProject: 'Déplacer vers un projet',
      movedTo: name => `Déplacée vers ${name}`,
      moveFailed: 'Impossible de déplacer la session',
      moveNoFolder: "Ce projet n'a aucun dossier vers lequel effectuer le déplacement",
      moveNoProjects: 'Aucun autre projet',
      reveal: 'Afficher dans le dossier',
      copyPath: 'Copier le chemin',
      removeFromSidebar: 'Masquer de la barre latérale',
      createFailed: 'Impossible de créer le projet',
      staleBackend:
        'Mettez à jour le backend Hermes pour créer des projets — votre backend est plus ancien que cette application de bureau (Paramètres → Mises à jour → Backend).',
      deleteConfirm:
        'Cela supprime le projet enregistré de Hermes. Les fichiers, dépôts git et worktrees restent inchangés.',
      startWork: 'Nouveau worktree',
      newWorktreeTitle: 'Nouveau worktree',
      newWorktreeDesc: 'Nommez la branche pour ce worktree.',
      branchPlaceholder: 'ex. ma-fonctionnalité',
      branchOff: () => ({ after: '', before: 'à partir de ' }),
      baseBranchPlaceholder: 'Rechercher des branches…',
      baseBranchNone: 'Aucune branche trouvée',
      startWorkFailed: 'Impossible de créer le worktree',
      worktreeStaleBackend:
        'Mettez à jour le backend Hermes pour créer des worktrees depuis Desktop — votre backend est plus ancien que cette application (Paramètres → Mises à jour → Backend).',
      worktreeProjectLabel: 'Projet',
      worktreeProjectPlaceholder: 'Rechercher des projets…',
      worktreeProjectNone: 'Aucun projet avec un dossier',
      convertBranch: 'Convertir une branche…',
      convertBranchTitle: 'Convertir une branche',
      convertBranchDesc: 'Ouvrez des branches vérifiées ou créez un worktree pour une branche libre.',
      convertBranchPlaceholder: 'Rechercher des branches…',
      convertBranchInstead: 'Convertir une branche existante',
      branchOpenExisting: 'ouverte',
      branchSwitchHome: "basculer l'accueil",
      branchCreateWorktree: 'nouveau worktree',
      branchTrackRemote: 'suivre la branche distante',
      branchesLoading: 'Chargement des branches…',
      noBranches: 'Aucune branche trouvée',
      removeWorktree: 'Supprimer le worktree',
      removeWorktreeFailed: 'Impossible de supprimer le worktree (modifications non validées ?)',
      removeWorktreeConfirm:
        'Supprimez-le de git (supprime le répertoire du worktree ; la branche reste), ou masquez simplement la voie de la barre latérale et laissez le worktree sur le disque.',
      removeWorktreeDirty:
        'Ce worktree a des modifications non validées. Forcez sa suppression (ces modifications seront perdues), ou masquez simplement la voie et conservez-le sur le disque.',
      forceRemove: 'Supprimer par force',
      enter: label => `Ouvrir ${label}`,
      reorder: label => `Réorganiser ${label}`,
      toggle: (label, open) => `${open ? 'Afficher' : 'Masquer'} les sessions de ${label}`,
      showAllCount: count => `Afficher les ${count} sessions`,
      back: 'Tous les projets'
    },
    newSessionIn: label => `Nouvelle session dans ${label}`,
    showMoreIn: (count, label) => `Afficher ${count} de plus dans ${label}`,
    loading: 'Chargement…',
    loadMore: 'Charger plus',
    loadCount: step => `Charger ${step} de plus`,
    messageCount: count => `${count} ${count === 1 ? 'message' : 'messages'}`,
    toolCallCount: count => `${count} ${count === 1 ? "appel d'outil" : "appels d'outil"}`,
    row: {
      pin: 'Épingler',
      unpin: 'Désépingler',
      markUnread: 'Marquer comme non lue',
      markRead: 'Marquer comme lue',
      unreadFailed: "Impossible de mettre à jour l'état de lecture",
      copyId: "Copier l'ID",
      export: 'Exporter',
      branchFrom: 'Branche',
      rename: 'Renommer',
      archive: 'Archiver',
      unarchive: 'Restaurer',
      newWindow: 'Nouvelle fenêtre',
      openInTerminal: 'Ouvrir dans le terminal',
      hideTabBar: "Masquer la barre d'onglets",
      openInNewTab: 'Ouvrir dans un nouvel onglet',
      openInSplit: 'Ouvrir en split',
      copyIdFailed: "Impossible de copier l'ID de session",
      sessionActions: 'Actions de session',
      sessionRunning: 'Session en cours',
      needsInput: 'Nécessite votre saisie',
      waitingForAnswer: 'En attente de votre réponse',
      finishedUnread: 'Terminée — non lue',
      backgroundRunning: 'Tâche en arrière-plan en cours',
      draftSession: 'Brouillon — aucun message envoyé',
      handoffOrigin: platform => `Transférée depuis ${platform}`,
      ownedByProfile: profile => `Profil : ${profile}`,
      renamed: 'Renommée',
      renameFailed: 'Échec du renommage',
      renameTitle: 'Renommer la session',
      renameDesc: 'Laissez vide pour effacer.',
      untitledPlaceholder: 'Session sans titre',
      deleteTitle: 'Supprimer la session ?',
      deleteDesc: title => `« ${title} » sera définitivement supprimée. Cette action est irréversible.`,
      deleting: 'Suppression…',
      deleted: 'Session supprimée',
      untitledChat: id => `Conversation ${id}`,
      messageCount: count => `${count} message${count === 1 ? '' : 's'}`,
      todoProgress: 'Tâches terminées',
      ageNow: 'maintenant',
      ageDay: 'j',
      ageHour: 'h',
      ageMin: 'm'
    },
    dateDivider: {
      today: "Aujourd'hui",
      yesterday: 'Hier',
      thisWeek: 'Plus tôt cette semaine',
      lastWeek: 'La semaine dernière',
      thisMonth: 'Plus tôt ce mois'
    },
    statusDivider: {
      working: 'En cours',
      done: 'Terminées'
    },
    markAllRead: 'Tout marquer comme lu'
  },
  composer: {
    message: 'Message',
    wakingProfile: profile => `Réveil de ${profile}…`,
    placeholderStarting: 'Démarrage de Hermes…',
    placeholderReconnecting: 'Reconnexion à Hermes…',
    placeholderFollowUp: 'Envoyer un suivi',
    newSessionPlaceholders: [
      'Sur quoi travaillons-nous ?',
      'Donnez une tâche à Hermes',
      "Qu'avez-vous en tête ?",
      'Décrivez ce dont vous avez besoin',
      "Qu'est-ce qu'on attaque ?",
      "Posez n'importe quelle question",
      'Commencez par un objectif'
    ],
    followUpPlaceholders: [
      'Envoyez un suivi',
      'Ajoutez du contexte',
      'Affinez la demande',
      'Et ensuite ?',
      'Continuez',
      'Poussez plus loin',
      'Ajustez ou continuez'
    ],
    startVoice: 'Démarrer la conversation vocale',
    openDirective: 'Ouvrir',
    queueMessage: "Mettre le message en file d'attente",
    steer: "Diriger l'exécution en cours",
    stop: 'Arrêter',
    send: 'Envoyer',
    speaking: 'Parle',
    transcribing: 'Transcription',
    thinking: 'Réfléchit',
    muted: 'Coupe',
    listening: 'Écoute',
    muteMic: 'Couper le microphone',
    unmuteMic: 'Réactiver le microphone',
    stopListening: "Arrêter l'écoute et envoyer",
    stopShort: 'Arrêter',
    endConversation: 'Terminer la conversation vocale',
    endShort: 'Terminer',
    stopDictation: 'Arrêter la dictée',
    transcribingDictation: 'Transcription de la dictée',
    voiceControls: 'Voix',
    voiceEngine: 'Moteur de conversation vocale',
    voiceEngineChained: 'Reconnaissance vocale + voix Hermes',
    voiceEngineLive: 'GPT-Live (duplex intégral, délègue à Hermes)',
    voiceEngineLiveNeedsKey: 'Nécessite une clé API OpenAI',
    voiceEngineChangeFailed: 'Impossible de changer le moteur de conversation vocale',
    voiceEngineChainedShort: 'reconnaissance vocale',
    voiceEngineLiveShort: 'GPT-Live',
    voiceDictation: 'Dictée vocale',
    speakReplies: 'Lire les réponses à haute voix',
    stopSpeakingReplies: 'Arrêter la lecture des réponses à haute voix',
    wakeWord: phrase => `Mot d'activation « ${phrase} »`,
    wakeWordListening: phrase => `Mot d'activation : « ${phrase} » — écoute en cours`,
    wakeWordOff: phrase => `Mot d'activation : « ${phrase} » — désactivé`,
    wakeWordPausedVoice: phrase => `Mot d'activation : « ${phrase} » — en pause pendant la conversation vocale`,
    lookupLoading: 'Recherche…',
    lookupNoMatches: 'Aucune correspondance.',
    lookupTry: 'Essayez',
    lookupOr: 'ou',
    commonCommands: 'Commandes courantes',
    hotkeys: 'Raccourcis clavier',
    helpFooter: 'ouvre le panneau complet · effacer pour fermer',
    commandDescs: {
      '/help': 'Afficher les commandes slash du bureau',
      '/clear': 'démarrer une nouvelle session',
      '/resume': 'Reprendre une session enregistrée',
      '/details': 'contrôler le niveau de détail de la transcription',
      '/copy': "copier la sélection ou le dernier message de l'assistant",
      '/quit': 'quitter Hermes',
      '/start': 'Accuser réception des pings de démarrage de la plateforme sans répondre',
      '/new': 'Démarrer une nouvelle conversation',
      '/topic': 'Activer ou inspecter les sessions par sujet des MP Telegram',
      '/save': 'Enregistrer la transcription actuelle en JSON',
      '/retry': 'Réessayer le dernier message (le renvoyer à l’agent)',
      '/prompt': 'Rédiger votre prochain prompt dans $EDITOR (markdown), puis l’envoyer',
      '/undo': 'Revenir de N tours utilisateur et relancer (1 par défaut)',
      '/title': 'Renommer la session actuelle',
      '/handoff': 'Transférer cette session vers une plateforme de messagerie',
      '/branch': 'Dériver le dernier message dans une nouvelle conversation',
      '/worktree': 'Afficher, lister, créer ou élaguer des worktrees git isolés',
      '/compress': 'Compresser le contexte de cette conversation',
      '/rollback':
        'Lister ou restaurer les points de contrôle du système de fichiers (les restaurations conservent vos modifications manuelles ; --all les écrase)',
      '/export': 'Exporter un profil (configuration, skills, thème) dans une archive partageable',
      '/import': 'Importer une archive de profil partagée comme nouveau profil',
      '/stop': 'Arrêter le tour actif et les processus en arrière-plan',
      '/pause': "Suspendre globalement tout nouveau travail (arrêt d’urgence) ; '/pause off' reprend",
      '/bg': 'Exécuter un prompt dans une session d’arrière-plan distincte',
      '/btw': 'Poser une question annexe sur cette conversation sans l’interrompre',
      '/agents': 'Afficher les agents actifs et les tâches en cours',
      '/journey': 'Ouvrir le graphe de mémoire — skills et souvenirs au fil du temps',
      '/queue':
        'Mettre un prompt en file pour le prochain tour, ou lister/modifier/supprimer/déplacer/vider les prompts en file',
      '/steer': 'Injecter un message après le prochain appel d’outil sans interrompre',
      '/goal':
        'Définir un objectif permanent sur lequel Hermes travaille au fil des tours jusqu’à ce qu’il soit atteint',
      '/heartbeat': 'Définir un prompt récurrent qui revient dans cette session lorsqu’elle est inactive',
      '/refine': 'Passer en revue cette conversation maintenant et enregistrer les leçons en mémoire/skills',
      '/review': 'Lancer un sous-agent indépendant pour relire le travail qui vient d’être discuté (PR, code, docs)',
      '/loop': 'Relancer un prompt à intervalle régulier dans cette session',
      '/plan': 'Écrire un plan d’implémentation markdown dans .hermes/plans/ sans rien exécuter',
      '/moa': 'Exécuter un prompt avec le préréglage Mixture of Agents par défaut, puis restaurer votre modèle',
      '/subgoal': 'Ajouter ou gérer des critères supplémentaires sur l’objectif actif',
      '/status': 'Afficher l’état de la session actuelle',
      '/egress': 'Afficher l’état du proxy de sortie Docker',
      '/context':
        'Afficher la vue détaillée de la fenêtre de contexte avec jauge d’utilisation, répartition par catégorie, statistiques de compression et débit',
      '/whoami': 'Afficher votre accès aux commandes slash (admin / utilisateur)',
      '/profile': 'Changer de profil Hermes actif',
      '/codex-runtime': 'Activer/désactiver le runtime codex app-server pour les modèles OpenAI/Codex',
      '/personality': 'Définir une personnalité prédéfinie',
      '/battery': 'Afficher/masquer un indicateur de batterie coloré dans la barre d’état',
      '/timestamps': 'Afficher/masquer les horodatages [HH:MM] sur les messages et /history',
      '/diff': 'Afficher les modifications git dans le répertoire de travail',
      '/focus': 'Activer/désactiver la vue focus — n’afficher que votre prompt et la réponse finale',
      '/yolo': 'Activer/désactiver YOLO — approuver automatiquement les commandes dangereuses',
      '/approvals': 'Afficher ou définir le mode d’approbation persistant des commandes dangereuses',
      '/reasoning': 'Effort ou affichage du raisonnement [<level> [--global]|show|hide|full|clamp]',
      '/skin': 'Changer de thème de bureau ou passer au suivant',
      '/wake': 'Contrôler l’écoute du mot d’activation du bureau [on|off|status]',
      '/tools': 'Gérer les outils : /tools [list|disable|enable] [name...]',
      '/memory': 'Examiner les écritures mémoire en attente / activer ou désactiver la validation',
      '/bundles': 'Lister les lots de skills (alias /<name> pour plusieurs skills)',
      '/pet': 'Afficher/masquer ou adopter une mascotte petdex (/pet, /pet list, /pet boba)',
      '/hatch': 'Générer une nouvelle mascotte (ouvre le générateur)',
      '/learn':
        'Apprendre une skill réutilisable à partir de ce que vous décrivez (dossiers, URL, cette conversation, notes)',
      '/init': 'Générer ou mettre à jour les instructions de projet AGENTS.md à partir d’une analyse du dépôt',
      '/suggestions': 'Examiner les automatisations suggérées (accepter/ignorer)',
      '/blueprint': 'Configurer une automatisation à partir d’un modèle',
      '/browser': 'Gérer la connexion CDP du navigateur [connect|disconnect|status] (gateway local uniquement)',
      '/palette': 'Ouvrir la palette de commandes floue (aussi Ctrl+P)',
      '/usage':
        'Afficher l’utilisation des jetons et les limites de débit ; `reset` utilise une réinitialisation de limite Codex en réserve',
      '/subscription': 'Voir votre forfait Nous et le modifier dans le navigateur',
      '/topup': 'Afficher votre solde Nous et gérer la facturation sur le portail',
      '/platform': 'Suspendre, reprendre ou lister une plateforme de gateway en échec',
      '/version': 'Afficher la version de Hermes Agent',
      '/debug': 'Téléverser un rapport de débogage (infos système + journaux) et obtenir des liens partageables',
      '/model': 'Changer le modèle de cette session'
    },
    hotkeyDescs: {
      'composer.mention': 'référencer des fichiers, dossiers, URLs, git',
      'composer.slash': 'palette de commandes slash',
      'composer.help': 'aide rapide (effacer pour fermer)',
      'composer.sendNewline': 'envoyer · Shift+Entrée pour un retour à la ligne',
      'composer.sendQueued': "envoyer le prochain tour en file d'attente",
      'keybinds.openPanel': 'tous les raccourcis clavier',
      'composer.cancel': "fermer le popover · annuler l'exécution",
      'composer.history': 'parcourir le popover / historique'
    },
    attachUrlTitle: 'Attacher une URL',
    attachUrlDesc: "Hermes récupérera la page et l'inclura comme contexte pour ce tour.",
    urlPlaceholder: 'https://example.com/post',
    urlHintPre: "Incluez l'URL complète, par ex. ",
    attach: 'Attacher',
    queued: count => `${count} en file d'attente`,
    queuedPaused: count => `${count} en file d'attente — en pause`,
    attachmentOnly: 'Tour avec seulement des pièces jointes',
    emptyTurn: 'Tour vide',
    hiddenQueued: 'Note de configuration',
    attachments: count => `${count} ${count === 1 ? 'pièce jointe' : 'pièces jointes'}`,
    editingInComposer: 'Modification dans le compositeur',
    editingQueuedInComposer: "Modification du tour en file d'attente dans le compositeur",
    restoredDraftNotice: 'Votre message non envoyé a été restauré',
    restoredDraftUndo: 'Annuler',
    queueEdit: 'Modifier',
    queueSendNext: 'Suivant',
    queueSteer: 'Diriger — réorienter maintenant le tour en cours',
    queueSend: 'Envoyer',
    queueDelete: 'Supprimer',
    queueResume: 'Reprendre',
    queueResumeTip: "Mis en pause par Arrêter — reprendre l'envoi des tours en file d'attente",
    queueStuckTitle: "Message en file d'attente non envoyé",
    queueStuckBody:
      "Un tour en file d'attente a continué à échouer lors de l'envoi. Il est toujours en file d'attente — essayez de l'envoyer à nouveau.",
    previewUnavailable: 'Aperçu indisponible',
    previewLabel: label => `Aperçu ${label}`,
    couldNotPreview: label => `Impossible d'apercevoir ${label}`,
    removeAttachment: label => `Supprimer ${label}`,
    dictating: 'Dictée',
    preparingAudio: "Préparation de l'audio",
    speakingResponse: 'Lecture de la réponse',
    readingAloud: 'Lecture à haute voix',
    themeSuggestions: 'Suggestions de thème du bureau',
    noMatchingThemes: 'Aucun thème correspondant.',
    themeTryPre: 'Essayez ',
    themeTryPost: '.',
    attachLabel: 'Attacher',
    files: 'Fichiers…',
    folder: 'Dossier…',
    images: 'Images…',
    pasteImage: 'Coller une image',
    url: 'URL…',
    promptSnippets: "Extraits d'invite…",
    tipPre: 'Conseil : tapez ',
    tipPost: ' pour référencer des fichiers en ligne.',
    snippetsTitle: "Extraits d'invite",
    snippetsDesc: 'Choisissez une invite de démarrage à insérer dans le compositeur.',
    dropFiles: 'Déposez des fichiers pour les attacher',
    dropSession: 'Déposez pour lier cette conversation',
    mcpSuggestions: {
      label: server => `Ajouter ${server}`,
      tip: keyword => `Suggéré car vous avez mentionné « ${keyword} » — cliquez pour vous connecter`,
      connecting: server => `Connexion à ${server}…`,
      cancelTip: 'Cliquez pour annuler',
      added: server => `${server} ajouté`,
      addedTip: 'Connecté — ses outils sont prêts dans cette conversation',
      connectFailed: server => `Impossible de se connecter à ${server}`
    },
    skillSuggestions: {
      label: skill => `Utiliser le skill : ${skill}`,
      tip: skill => `Vous avez mentionné « ${skill} » — cliquez pour commencer avec ce skill`,
      done: skill => `/${skill} ajouté`,
      doneTip: "Le skill se charge lors de l'envoi"
    },
    githubSuggestions: {
      label: 'Configurer GitHub',
      tip: 'GitHub fonctionne ici grâce aux skills de la CLI gh ; cliquez pour connecter votre compte',
      done: 'Ajout de /github-auth',
      doneTip: "Envoyez le message et l'agent vous guidera dans la connexion à GitHub"
    },
    repairSuggestions: {
      label: server => `Reconnecter ${server}`,
      tip: server => `Un appel à ${server} vient d'échouer à cause d'une erreur de connexion`,
      working: server => `Reconnexion à ${server}…`,
      workingTip: 'Cliquez pour annuler',
      done: server => `${server} reconnecté`,
      doneTip: 'Les nouveaux identifiants sont actifs dans cette conversation',
      failed: server => `Impossible de reconnecter ${server}`
    },
    cronSuggestions: {
      label: 'Planifier ceci',
      tip: phrase => `« ${phrase} » semble récurrent — exécutez-le plutôt selon une planification`,
      prefix: 'Configurer ceci comme tâche planifiée :',
      done: 'Marqué pour planification',
      doneTip: "Envoyez-le pour que l'agent crée la tâche"
    },
    snippets: {
      codeReview: {
        label: 'Revue de code',
        description:
          'Audit des modifications actuelles à la recherche de régressions, de cas limites oubliés et de tests manquants.',
        text: 'Veuillez examiner cela à la recherche de bugs, de régressions et de tests manquants.'
      },
      implementationPlan: {
        label: "Plan d'implémentation",
        description: 'Élaborez une approche avant de toucher au code pour que la différence reste ciblée.',
        text: "Veuillez élaborer un plan d'implémentation concis avant de modifier le code."
      },
      explainThis: {
        label: 'Expliquez ceci',
        description: 'Parcourez comment le code sélectionné fonctionne et liez les fichiers clés.',
        text: 'Veuillez expliquer comment cela fonctionne et pointez-moi vers les fichiers clés.'
      }
    }
  },
  statusStack: {
    hideStack: 'Masquer la pile d’état',
    showStack: 'Afficher la pile d’état',
    agents: 'Agents',
    background: count => `${count} arrière-plan`,
    goalActive: 'Objectif actif',
    goalBlocked: 'Objectif bloqué',
    goalDone: 'Objectif terminé',
    goalPaused: 'Objectif en pause',
    goalWaiting: 'Objectif en attente',
    subagents: count => `${count} sous-agent${count === 1 ? '' : 's'}`,
    todos: (done, total) => `Tâches ${done}/${total}`,
    running: 'En cours',
    stop: 'Arrêter',
    dismiss: 'Rejeter',
    exit: code => `sortie ${code}`,
    control: {
      goalActiveTurns: (turn, maxTurns) => `Tour ${turn}/${maxTurns}`,
      goalDoneTurns: turns => `${turns} tour${turns === 1 ? '' : 's'}`,
      goalTurn: turn => `Tour ${turn}`,
      goalActions: "Actions de l'objectif",
      viewDetails: 'Voir les détails',
      addCriterion: 'Ajouter un critère',
      addCriterionDialogTitle: 'Ajouter un critère',
      addCriterionPlaceholder: 'Saisissez le critère...',
      criterionLabel: 'Critère',
      pauseGoal: "Mettre l'objectif en pause",
      resumeGoal: "Reprendre l'objectif",
      resumeNow: 'Reprendre maintenant',
      clearGoal: "Supprimer l'objectif",
      clearGoalConfirmTitle: "Supprimer l'objectif ?",
      clearGoalConfirmBody: "Voulez-vous vraiment supprimer l'objectif actif ? Cette action est irréversible.",
      copyCriterion: index => `Copier le critère ${index}`,
      removeCriterion: index => `Supprimer le critère ${index}`,
      removeCriterionConfirmTitle: index => `Supprimer le critère ${index} ?`,
      removeCriterionConfirmBody: index => `Voulez-vous vraiment supprimer le critère ${index} ?`,
      clearCriteria: 'Supprimer tous les critères',
      clearCriteriaConfirmTitle: 'Supprimer tous les critères ?',
      clearCriteriaConfirmBody: 'Voulez-vous vraiment supprimer tous les critères de cet objectif ?',
      criteriaHeader: count => `Critères · ${count}`,
      noCriteria: 'Aucun critère',
      goalDetailsTitle: "Détails de l'objectif",
      objectiveLabel: 'Objectif',
      contractOutcome: 'Résultat attendu',
      contractVerification: 'Vérification',
      contractConstraints: 'Contraintes',
      contractBoundaries: 'Limites',
      contractStopWhen: "Condition d'arrêt",
      waitBarrierTitle: "Condition d'attente",
      waitUntil: target => `En attente jusqu'à ${target}`,
      waitSession: target => `En attente de la session ${target}`,
      waitPid: pid => `En attente du processus ${pid}`,
      qualityGatesTitle: 'Contrôles qualité',
      gateCommand: 'Commande',
      gateAttempts: (attempts, max) => `${attempts}/${max} tentatives`,
      gateTimeout: seconds => `Délai maximal : ${seconds} s`,
      gateLastExit: code => (code === null ? 'En attente' : `Code de sortie : ${code}`),
      loopActive: 'Boucle active',
      loopPaused: 'Boucle en pause',
      loopDeferred: 'Boucle différée',
      loopFinished: 'Boucle terminée',
      loopRuns: runs => `${runs} exécution${runs === 1 ? '' : 's'}`,
      loopRunCount: (current, total) => `Exécution ${current}/${total}`,
      loopNext: time => `prochaine : ${time}`,
      loopEverySeconds: seconds => `toutes les ${seconds} s`,
      loopEveryMinutes: minutes => `toutes les ${minutes} min`,
      loopEveryHours: hours => `toutes les ${hours} h`,
      loopSelfPaced: 'rythme autonome',
      loopActions: 'Actions de la boucle',
      pauseLoop: 'Mettre la boucle en pause',
      resumeLoop: 'Reprendre la boucle',
      stopLoop: 'Arrêter la boucle',
      stopLoopConfirmTitle: 'Arrêter la boucle ?',
      stopLoopConfirmBody: 'Voulez-vous vraiment arrêter cette boucle ?',
      dismissLoop: 'Masquer la boucle',
      loopPromptLabel: 'Consigne',
      loopCadenceLabel: 'Fréquence',
      loopUntilLabel: "Condition d'arrêt",
      loopDeferredNotice: 'Un objectif actif contrôle actuellement la session.',
      loopAwaitingResponse: "En attente d'une réponse",
      heartbeatActive: 'Suivi périodique actif',
      heartbeatPaused: 'Suivi périodique en pause',
      heartbeatEveryMinutes: minutes => `toutes les ${minutes} min`,
      heartbeatEveryHours: hours => `toutes les ${hours} h`,
      heartbeatEverySeconds: seconds => `toutes les ${seconds} s`,
      heartbeatNext: time => `prochain : ${time}`,
      heartbeatDueWaitingForIdle: 'échu — en attente de disponibilité',
      heartbeatActions: 'Actions du suivi périodique',
      pauseHeartbeat: 'Mettre le suivi en pause',
      resumeHeartbeat: 'Reprendre le suivi',
      clearHeartbeat: 'Supprimer le suivi',
      clearHeartbeatConfirmTitle: 'Supprimer le suivi périodique ?',
      clearHeartbeatConfirmBody: 'Voulez-vous vraiment supprimer ce suivi périodique ?',
      heartbeatFiredCount: count => `Déclenché ${count} fois`,
      actionFailed: msg => `Échec de l'action : ${msg}`,
      actionSucceeded: 'Action réussie',
      copySuccess: 'Critère copié dans le presse-papiers',
      copyFailure: 'Impossible de copier le critère dans le presse-papiers',
      continuationFailed: "Impossible de soumettre la poursuite de l'objectif",
      continuationQueued: 'Objectif repris — poursuite en attente de la fin du tour actuel',
      continuationBusy: 'Objectif repris — session occupée, utilisez /interrupt pour poursuivre',
      controlUnavailable: msg => `Commandes de session indisponibles : ${msg}`,
      dismissError: "Masquer l'erreur",
      add: 'Ajouter'
    },
    coding: {
      title: 'Arbre de travail',
      noBranch: 'Aucune branche',
      detached: 'détaché',
      clean: 'Propre',
      changed: count => `${count} ${count === 1 ? 'modifié' : 'modifiés'}`,
      ahead: count => `${count} en avance`,
      behind: count => `${count} en retard`,
      review: 'Revoir',
      close: 'Fermer',
      openChanges: 'Ouvrir les modifications',
      openFile: 'Ouvrir le fichier',
      stage: 'Mettre en zone de préparation',
      unstage: 'Retirer de la zone de préparation',
      stageAll: 'Tout mettre en zone de préparation',
      viewAsTree: 'Voir en arbre',
      viewAsList: 'Voir en liste',
      revert: 'Rétablir',
      revertAll: 'Tout rétablir',
      revertConfirm:
        "Abandonner les modifications de ce fichier et le rétablir à l'état validé ? Cela ne peut pas être annulé.",
      revertAllConfirm:
        "Abandonner toutes les modifications et rétablir les fichiers à l'état validé ? Cela ne peut pas être annulé.",
      staged: 'En zone de préparation',
      noChanges: 'Aucune modification',
      notRepo: 'Pas un dépôt git',
      noDiff: 'Aucune différence à afficher',
      scopeUncommitted: 'Non validé',
      scopeBranch: 'Branche',
      scopeLastTurn: 'Dernier tour',
      commit: 'Valider',
      commitAndPush: 'Valider et pousser',
      commitPlaceholder: shortcut => `Message (${shortcut} pour valider)`,
      generateCommitMessage: 'Générer un message de validation',
      stopGenerating: 'Arrêter la génération',
      createPr: 'Créer une PR',
      openPr: 'Ouvrir une PR',
      ghMissing: 'Installez la CLI GitHub (gh) et connectez-vous pour ouvrir des PR',
      agentShip: "Demander à Hermes d'ouvrir une PR",
      agentShipUnavailable: "La conversation à l'origine de ces modifications n'est pas affichée.",
      agentShipPrompt:
        'Passez en revue les modifications actuelles, validez-les avec un message de validation conventionnel clair, poussez la branche, puis ouvrez une pull request.',
      newBranch: 'Nouvelle branche',
      branchOffFrom: base => `Nouvelle branche à partir de ${base}`,
      switchTo: branch => `Basculer vers ${branch}`,
      switchFailed: branch => `Impossible de basculer vers ${branch}`,
      worktrees: 'Worktrees'
    }
  },
  updates: {
    discontinuedTitle: "Cette version de Hermes n'est plus prise en charge",
    discontinuedBody: "Cette version de Hermes n'est plus prise en charge et risque de ne plus fonctionner — désinstallez-la. Vos données restent sur le disque.",
    channels: { stable: 'Stable', canary: 'Canary' },
    appName: 'Hermes',
    availableBodyRelease: tag => `La version ${tag} est prête à être installée.`,
    releaseAvailable: tag => `La version ${tag} est disponible.`,
    checkingShort: 'Vérification…',
    availableBodyAppInstaller: "Une nouvelle version de Hermes est prête. Hermes va se fermer, Windows terminera la mise à jour, puis Hermes rouvrira automatiquement.",
    applyingBodyAppInstaller: "Hermes va se fermer et Windows terminera la mise à jour. Hermes rouvrira ensuite automatiquement — vous n'avez rien à faire.",
    applyingCloseAppInstaller: 'Cette fenêtre va se fermer, Windows terminera la mise à jour et Hermes rouvrira automatiquement.',
    checkUnknownTitleAppInstaller: 'Impossible de vérifier les mises à jour',
    checkUnknownBodyAppInstaller: "Windows n'a pas pu rechercher les mises à jour. Elles s'installent également automatiquement au redémarrage de Hermes.",
    versionDetailsTitle: 'Détails de la version',
    versionDetailsBody: "Cette installation est gérée hors de l'application. Mettez-la à jour de la même manière que vous l'avez installée.",
    versionDetailsVersion: 'Version',
    versionDetailsCommit: 'Commit',
    versionDetailsBuildOrigin: 'Origine de la compilation',
    versionDetailsDistribution: 'Distribution',
    versionDetailsDistributionDesktop: 'Application Desktop',
    versionDetailsDistributionDesktopMsix: 'Application Desktop (MSIX)',
    versionDetailsDistributionDesktopInstaller: 'Application Desktop (installateur)',
    versionDetailsDistributionSourceInstaller: "Code source (script d'installation)",
    versionDetailsDistributionSourceInstallerDesktop: "Code source (script d'installation) + hermes desktop",
    versionDetailsDistributionSource: 'Code source',
    versionDetailsDistributionSourceDesktop: 'Code source + hermes desktop',
    versionDetailsDistributionStore: 'Microsoft Store',
    versionDetailsRuntime: "Environnement d'exécution",
    versionDetailsRuntimeEmbedded: "Environnement d'exécution intégré",
    versionDetailsRuntimeExternal: "Externe (utilise l'environnement d'exécution du système)",
    versionDetailsInstallId: "ID d'installation",
    versionDetailsUncommittedChanges: 'modifications non commitées',
    version: value => `Version ${value}`,
    versionUnavailable: 'Version indisponible',
    bundleOutOfSync: "Version de l'application obsolète",
    bundleOutOfSyncDesc:
        "Le runtime Hermes a été mis à jour, mais l'application Desktop utilise encore une ancienne version. Les nouvelles fonctions de l'interface, comme le mode Bot, resteront absentes jusqu'à sa mise à jour. Lancez la mise à jour ci-dessous pour reconstruire l'application. Si cet avertissement persiste, réinstallez-la avec le dernier installateur Desktop.",
    bundleOutOfSyncAction: "Obtenir l'installateur",
    bundleSwapPending: 'Redémarrez pour terminer la mise à jour',
    bundleSwapPendingDesc:
        "L'application mise à jour est déjà installée — Hermes doit seulement redémarrer pour la charger. Vos conversations et paramètres sont préservés.",
    bundleSwapPendingAction: 'Redémarrer Hermes',
    checkNow: 'Vérifier maintenant',
    seeWhatsNew: 'Voir les nouveautés',
    releaseNotes: 'Notes de version',
    onLatest: 'Vous utilisez la dernière version.',
    installing: "Une mise à jour est en cours d'installation.",
    cantReach: "Impossible d'atteindre le serveur de mises à jour.",
    tapCheck: 'Cliquez sur « Vérifier maintenant » pour rechercher des mises à jour.',
    updateReady: count => `Une nouvelle mise à jour est prête (${count} changement${count === 1 ? '' : 's'} inclus).`,
    updateReadyUnknown: 'Une nouvelle mise à jour est prête.',
    lastChecked: age => `Dernière vérification ${age}`,
    justNowSuffix: " · à l'instant",
    never: 'jamais',
    justNow: "à l'instant",
    minAgo: count => `il y a ${count} min`,
    hoursAgo: count => `il y a ${count} h`,
    daysAgo: count => `il y a ${count} j`,
    stages: {
      idle: 'Préparation…',
      prepare: 'Préparation…',
      fetch: 'Téléchargement…',
      pull: 'Presque prêt…',
      pydeps: 'Finalisation…',
      update: 'Mise à jour de Hermes…',
      rebuild: "Reconstruction de l'application de bureau…",
      restart: 'Redémarrage de Hermes…',
      done: 'Mise à jour terminée',
      manual: 'Mise à jour depuis votre terminal',
      guiSkew: "Mettre à jour l'application de bureau",
      error: 'Mise à jour en pause'
    },
    checking: 'Recherche de mises à jour…',
    checkFailedTitle: 'Impossible de vérifier les mises à jour',
    tryAgain: 'Réessayer',
    notAvailableTitle: 'Mise à jour indisponible',
    unsupportedMessage: "Cette version de Hermes ne peut pas se mettre à jour depuis l'application.",
    connectionRetry: 'Vérifiez votre connexion et réessayez.',
    gitUnusable: 'Hermes n’a pas pu exécuter Git sur cet ordinateur et n’a donc pas pu rechercher de mises à jour.',
    connectionSettings: 'Paramètres de connexion',
    openDownloadPage: 'Ouvrir la page de téléchargement',
    latestBody: 'Vous utilisez la dernière version.',
    latestBodyBackend: 'Le backend utilise la dernière version.',
    allSetTitle: 'Tout est prêt',
    availableTitle: 'Nouvelle mise à jour disponible',
    availableBody: 'Une nouvelle version de Hermes est prête à être installée.',
    availableTitleBackend: 'Mise à jour du backend disponible',
    availableBodyBackend: 'Une version plus récente du backend Hermes connecté est prête à être installée.',
    availableBodyNoChangelog:
      "Une version plus récente est prête. Les notes de version ne sont pas disponibles pour ce type d'installation.",
    updateNow: 'Mettre à jour maintenant',
    maybeLater: 'Peut-être plus tard',
    moreChanges: count =>
      `+ ${count} ${count === 1 ? 'changement supplémentaire inclus' : 'changements supplémentaires inclus'}.`,
    manualTitle: 'Mise à jour depuis votre terminal',
    manualUnavailableTitle: 'Mise à jour impossible ici',
    manualBody:
      "Vous avez installé Hermes depuis la ligne de commande, les mises à jour s'y effectuent donc aussi. Collez ceci dans votre terminal :",
    manualPickedUp: 'Hermes prendra en compte la nouvelle version au prochain lancement.',
    manualBodyBackend: 'Le backend Hermes est géré en dehors de cette app. Exécutez ceci sur le serveur qui l’héberge :',
    manualPickedUpBackend: 'Le backend chargera la nouvelle version une fois la mise à jour terminée.',
    guiSkewTitle: "Mettre à jour l'application de bureau",
    guiSkewBody:
      "Le backend a été mis à jour, mais ce package d'application de bureau ne l'a pas été. Mettez à jour ou réinstallez l'application de bureau Hermes (votre AppImage / .deb / .rpm) pour qu'elle corresponde.",
    copy: 'Copier',
    copied: 'Copié',
    done: 'Terminé',
    applyingBody:
      'Le programme de mise à jour de Hermes prend le relais dans sa propre fenêtre et rouvre Hermes automatiquement une fois terminé. Ne rouvrez pas Hermes vous-même pendant la mise à jour.',
    applyingBodyBackend:
      'Le backend distant applique la mise à jour et va redémarrer. Hermes se reconnecte automatiquement à son retour.',
    applyingClose: 'Cette fenêtre se fermera pendant la mise à jour, puis Hermes se rouvre seul.',
    errorTitle: 'Mise à jour non terminée',
    errorBody: "Pas de souci — rien n'a été perdu. Vous pouvez réessayer maintenant.",
    blockerTitle: 'Fermer les aperçus locaux pour mettre à jour Hermes ?',
    blockerBody:
      'Hermes doit arrêter ces aperçus locaux avant la mise à jour. Aucun de vos fichiers ne sera modifié ni supprimé.',
    foreignBlockerTitle: 'Fermez les autres processus pour mettre à jour Hermes',
    foreignBlockerBody:
      "Hermes ne peut pas fermer automatiquement ces processus en toute sécurité. Fermez l'application, le terminal ou le service qui possède chacun d'eux, puis relancez la mise à jour.",
    mixedBlockerBody:
      'Hermes peut fermer les aperçus locaux ci-dessous. Les autres processus doivent être fermés manuellement avant de poursuivre la mise à jour.',
    closePreviewsAndUpdate: 'Fermer les aperçus et mettre à jour',
    closePreviewsAndCheckAgain: 'Fermer les aperçus et revérifier',
    localPreview: 'Aperçu local',
    portLabel: port => `Port ${port}`,
    pidLabel: pid => `PID ${pid}`,
    technicalDetails: 'Détails techniques',
    notNow: 'Pas maintenant',
    clientAlsoBehindTitle: "L'application Desktop n'est pas à jour",
    clientAlsoBehindMessage:
      'Le backend est à jour, mais cette application Desktop utilise encore une ancienne version. Mettez-la à jour pour profiter des derniers correctifs.',
    clientAlsoBehindAction: "Mettre à jour l'application Desktop",
    everythingDispatched: 'Mise à jour envoyée',
    everythingSkipped: 'Ignorée',
    everythingRowFailed: 'Échec de la mise à jour',
    everythingFanoutFailedTitle: 'Impossible de mettre à jour les autres instances',
    changeLogNew: 'Nouveautés',
    changeLogFixed: 'Corrigé',
    changeLogFaster: 'Plus rapide',
    changeLogImproved: 'Amélioré',
    changeLogOther: 'Autres améliorations',
    changeLogFallbackLabel: 'Dans cette mise à jour',
    changeLogFallbackItem: 'Améliorations et corrections',
    applyStatus: {
      preparing: 'Mise à jour du backend…',
      pulling: 'Mise à jour du backend…',
      restarting: 'Redémarrage du backend pour charger la mise à jour…',
      notAvailable: 'Mise à jour indisponible pour ce backend.',
      failed: 'Échec de la mise à jour du backend.',
      noReturn:
        "Le backend ne s'est pas reconnecté. La mise à jour n'est peut-être pas terminée — vérifiez l'hôte du backend."
    }
  },
  handoffTour: {
    profileTitle: 'Votre première tâche utilise le profil par défaut',
    profileText:
      "Cette barre change de profil. Celui qui est éclairé est le profil par défaut, où se trouve la session de la tâche. L'autre est le profil de configuration, où se trouve la conversation de bienvenue.",
    sessionsTitle: 'Chaque profil conserve ses propres sessions',
    sessionsText:
      'Cette liste appartient au profil par défaut. Nouvelle session démarre une tâche sur le profil sélectionné. Changez de profil dans la barre et la liste change avec lui.',
    stayTitle: "Hermes reste à portée d'un clic",
    stayText:
      'Passez au profil de configuration et ouvrez Bienvenue dans Hermes lorsque vous avez besoin d’aide. La conversation y reste disponible.'
  },
  guidedGreeting: {
    line: "Salut, entrez ! Je suis Hermes. Donnez-moi deux minutes pour préparer les lieux à votre façon, puis nous nous attaquerons à quelque chose que vous voulez vraiment accomplir.\n\nMais d'abord, comment dois-je vous appeler ?",
    nameSuggestion: name => `(Je peux aussi simplement vous appeler ${name}, si vous préférez.)`
  },
  install: {
    stageStates: {
      pending: 'En attente',
      running: 'Installation',
      succeeded: 'Terminé',
      skipped: 'Ignoré',
      failed: 'Échoué'
    },
    oneTimeTitle: 'Hermes nécessite une installation unique',
    unsupportedDesc: platform =>
      `L'installation automatisée au premier lancement n'est pas encore disponible sur ${platform}. Ouvrez le Terminal et exécutez la commande ci-dessous, puis relancez cette application. Les lancements suivants ignoreront cette étape.`,
    installCommand: "Commande d'installation",
    copyCommand: 'Copier la commande',
    viewDocs: "Voir la documentation d'installation",
    installTo: 'Sera installé dans',
    retryAfterRun: "Je l'ai exécuté — réessayer",
    setupChoiceTitle: 'Configurer Hermes Desktop',
    setupChoiceDesc:
      'Connectez cette application à un gateway Hermes que vous exécutez déjà, ou installez Hermes localement sur cet ordinateur.',
    connectExistingTitle: 'Se connecter à un Hermes existant',
    connectExistingShort: 'Connecter un existant',
    connectExistingDesc:
      'Utilisez un backend distant avec un jeton de session ou une connexion par navigateur. Aucune installation locale ne démarrera.',
    installLocalTitle: 'Installer Hermes localement',
    installLocalDesc: 'Téléchargez Hermes, créez son environnement Python et exécutez le backend sur cet ordinateur.',
    localStartUnavailable: "L'installation locale n'a pas pu démarrer. Redémarrez Hermes Desktop et réessayez.",
    remoteSetupTitle: 'Se connecter à un Hermes existant',
    remoteSetupDesc:
      "Entrez l'URL du gateway. Hermes Desktop détectera s'il a besoin d'un jeton ou d'une connexion par navigateur.",
    remoteUrlTitle: 'URL du gateway',
    remoteUrlDesc: "Utilisez l'URL de base du gateway Hermes, y compris https:// pour les connexions distantes.",
    remoteUrlPlaceholder: 'https://gateway.example.com/hermes',
    probing: "Détection de l'authentification du gateway...",
    probeError: "Impossible d'atteindre ce gateway Hermes.",
    probeErrorDetails: 'Détails',
    identityProvider: "votre fournisseur d'identité",
    authTitle: 'Authentification',
    authNeedsOauth: provider => `Connectez-vous avec ${provider} avant de tester ce gateway.`,
    authSignedIn: 'Connexion par navigateur terminée.',
    connected: 'Connecté',
    signIn: 'Se connecter',
    signInWith: provider => `Se connecter avec ${provider}`,
    enterUrlFirst: "Entrez d'abord une URL de gateway.",
    signInIncomplete: "La fenêtre de connexion s'est fermée avant la fin de l'authentification.",
    tokenTitle: 'Jeton de session',
    tokenDesc: 'Collez le jeton de session issu du fichier .env du gateway distant.',
    pasteSessionToken: 'Coller le jeton de session',
    incompleteSignInTest: 'Connectez-vous avant de tester ce gateway protégé par OAuth.',
    incompleteTokenTest: 'Entrez un jeton de session avant de tester ce gateway.',
    testConnection: 'Tester la connexion',
    testSucceeded: (baseUrl, version) => `Connecté à ${baseUrl}${version ? ` (${version})` : ''}.`,
    applyRemote: 'Appliquer et se reconnecter',
    backToSetup: 'Retour',
    failedTitle: "Échec de l'installation",
    settingUpTitle: 'Configuration de Hermes Agent',
    finishingTitle: 'Finalisation',
    failedDesc:
      "L'une des étapes d'installation a échoué. Sous Windows, cela peut arriver si une autre instance Hermes CLI ou desktop est en cours d'exécution. Arrêtez toutes les instances Hermes en cours, puis réessayez. Consultez les détails ci-dessous ou le journal du bureau pour la transcription complète.",
    activeDesc:
      "Il s'agit d'une configuration unique. Le programme d'installation de Hermes télécharge les dépendances et configure votre machine. Les lancements suivants ignoreront cette étape.",
    progress: (completed, total) => `${completed} sur ${total} étapes terminées`,
    currentStage: stage => ` -- actuellement : ${stage}`,
    fetchingManifest: "Récupération du manifeste d'installation...",
    error: 'Erreur',
    hideOutput: "Masquer la sortie de l'installateur",
    showOutput: "Afficher la sortie de l'installateur",
    lines: count => `${count} ligne${count === 1 ? '' : 's'}`,
    noOutput: 'Aucune sortie pour le moment.',
    cancelling: 'Annulation...',
    cancelInstall: "Annuler l'installation",
    transcriptSaved: 'Transcription complète enregistrée dans',
    copiedOutput: 'Copié !',
    copyOutput: 'Copier la sortie',
    reloadRetry: 'Recharger et réessayer',
    openLogs: 'Ouvrir les journaux'
  },
  onboarding: {
    headerTitle: 'Configurons Hermes Agent pour vous',
    headerDesc:
      'Connectez un fournisseur de modèles pour commencer à discuter. La plupart des options nécessitent un clic.',
    preparingInstall:
      "Hermes finalise l'installation. Cela prend généralement moins d'une minute au premier lancement.",
    starting: 'Démarrage de Hermes…',
    lookingUpProviders: 'Recherche des fournisseurs...',
    collapse: 'Réduire',
    otherProviders: 'Autres fournisseurs',
    haveApiKey: 'Vous avez une clé API ?',
    chooseLater: 'Je choisirai un fournisseur plus tard',
    recommended: 'Recommandé',
    connected: 'Connecté',
    featuredPitch: 'Un abonnement, 300+ modèles de pointe — la méthode recommandée pour exécuter Hermes',
    fireworksPitch: 'API de modèles directe — modèles de pointe hébergés par Fireworks',
    localModelsTitle: 'Exécuter des modèles en local',
    localModelsPitch: 'Aucun compte requis — téléchargez un modèle et exécutez-le sur cette machine',
    openRouterPitch: 'Une clé, des centaines de modèles — une valeur par défaut solide',
    apiKeyOptions: {
      fireworks: {
        short: 'API de modèles directe',
        description: 'Accès direct aux modèles hébergés par Fireworks AI.'
      },
      openrouter: {
        short: 'une clé, de nombreux modèles',
        description:
          'Héberge des centaines de modèles derrière une seule clé. Bonne valeur par défaut pour les nouvelles installations.'
      },
      openai: {
        short: 'Modèles de classe GPT',
        description: 'Accès direct aux modèles OpenAI.'
      },
      gemini: {
        short: 'Modèles Gemini',
        description: 'Accès direct aux modèles Google Gemini.'
      },
      xai: {
        short: 'Modèles Grok',
        description: 'Accès direct aux modèles xAI Grok.'
      },
      local: {
        short: 'auto-hébergé',
        description:
          'Pointez Hermes vers un point de terminaison local ou auto-hébergé compatible OpenAI (vLLM, llama.cpp, Ollama, etc).'
      }
    },
    backToSignIn: 'Retour à la connexion',
    getKey: 'Obtenir une clé',
    replaceCurrent: 'Remplacer la valeur actuelle',
    pasteApiKey: 'Collez votre clé API',
    localApiKeyPlaceholder: 'Clé API (facultatif — uniquement si votre point de terminaison en requiert une)',
    couldNotSave: "Impossible d'enregistrer l'identifiant.",
    connecting: 'Connexion',
    update: 'Mettre à jour',
    flowSubtitles: {
      pkce: 'Ouvre votre navigateur pour vous connecter, puis continue ici',
      device_code: 'Ouvre une page de vérification dans votre navigateur — Hermes se connecte automatiquement',
      external: 'Connectez-vous une fois dans votre terminal, puis revenez discuter'
    },
    startingSignIn: provider => `Démarrage de la connexion pour ${provider}...`,
    verifyingCode: provider => `Vérification de votre code avec ${provider}...`,
    connectedProvider: provider => `${provider} connecté`,
    connectedPicking: provider => `${provider} connecté. Choix d'un modèle par défaut...`,
    signInFailed: 'Échec de la connexion. Réessayez.',
    signInExpired:
      "La connexion a expiré dans l'attente de l'autorisation. Cela signifie généralement que la page de connexion s'est figée dans l'onglet ouvert (problème côté serveur) — terminez la connexion dans cet onglet, puis réessayez. Si le problème persiste, utilisez plutôt une clé API ou la solution de secours en ligne de commande.",
    signInDidNotFinish: provider =>
      `La connexion avec ${provider} ne s'est pas terminée. Vérifiez votre connexion Internet et réessayez, ou choisissez un autre fournisseur.`,
    tryAgain: 'Réessayer',
    useApiKeyInstead: 'Utiliser une clé API',
    errorDetails: 'Détails',
    pickDifferentProvider: 'Choisissez un autre fournisseur',
    signInWith: provider => `Se connecter avec ${provider}`,
    openedBrowser: provider => `Nous avons ouvert ${provider} dans votre navigateur.`,
    authorizeThere: 'Autorisez Hermes là-bas.',
    copyAuthCode: "Copiez le code d'autorisation et collez-le ci-dessous.",
    pasteAuthCode: "Coller le code d'autorisation",
    reopenAuthPage: "Rouvrir la page d'autorisation",
    autoBrowser: provider =>
      `Nous avons ouvert ${provider} dans votre navigateur. Autorisez Hermes là-bas et vous serez connecté automatiquement — rien à copier ou coller.`,
    reopenSignInPage: 'Rouvrir la page de connexion',
    waitingAuthorize: 'En attente de votre autorisation...',
    externalPending: provider =>
      `${provider} se connecte via sa propre CLI. Exécutez cette commande dans un terminal, puis revenez et choisissez « Je me suis connecté » :`,
    signedIn: 'Je me suis connecté',
    deviceCodeOpened: provider => `Nous avons ouvert ${provider} dans votre navigateur. Entrez ce code là-bas :`,
    reopenVerification: 'Rouvrir la page de vérification',
    copy: 'Copier',
    defaultModel: 'Modèle par défaut',
    freeTier: 'Gratuit',
    pro: 'Pro',
    free: 'Gratuit',
    price: (input, output) => `${input} entrant / ${output} sortant par Mtok`,
    change: 'Modifier',
    startChatting: 'Commencer',
    docs: provider => `Documentation ${provider}`
  },
  freeTier: {
    providerRowTitle: 'Nous · offre gratuite',
    providerRowPitch: 'Connectez-vous avec un compte Nous pour débloquer davantage de modèles et outils.',
    readyTitle: 'Hermes est prêt.',
    readyCaption: 'Gratuit · connecteurs inclus',
    begin: 'Commencer',
    signInInstead: 'Se connecter plutôt avec un compte Nous',
    otherProviders: 'Autres fournisseurs',
    stripTitle: "L'inférence Nous gratuite et les connecteurs sont maintenant disponibles.",
    stripBody: 'Ouvrez le sélecteur de modèle pour les essayer ou connectez-vous avec un compte Nous.',
    openModelPicker: 'Ouvrir le sélecteur de modèle',
    dismiss: 'Fermer',
    providerName: 'Nous',
    statusLabel: model => `Nous · ${model}`,
    signIn: 'Se connecter',
    signInHeading: 'Connectez-vous avec un compte Nous pour débloquer davantage de modèles et outils.',
    settingUp: "Configuration de l'inférence gratuite…",
    codeBody: 'Saisissez ce code dans votre navigateur pour terminer la connexion.',
    copyLink: 'Copier le lien',
    doNotShare: 'Ne partagez pas ce code.',
    waiting: 'En attente de la connexion…',
    finishingHeading: 'Finalisation de la connexion…',
    finishingBody: 'Autorisation accordée dans le navigateur. Récupération des jetons de votre compte.',
    signedInAs: email => `Connecté en tant que ${email}`,
    signedIn: 'Connecté.',
    completedBody: "Votre compte donne maintenant accès à l'inférence et aux outils.",
    defaultModel: 'Modèle par défaut',
    change: 'Modifier',
    done: 'Terminé',
    notNow: 'Pas maintenant',
    tryAgain: 'Réessayer',
    startAgain: 'Recommencer',
    didNotComplete: "La connexion n'a pas abouti",
    rejectedBody: "La connexion a été refusée dans le navigateur. Vous restez sur l'offre gratuite.",
    supersededBody: 'Un code de connexion plus récent a remplacé celui-ci.',
    timedOutHeading: 'Délai de connexion dépassé',
    timedOutBody: "Le code n'a pas été utilisé à temps. Vous restez sur l'offre gratuite.",
    retiredBody:
      "Cette identité d'offre gratuite a déjà été utilisée ou a expiré ; une nouvelle sera créée au prochain démarrage.",
    errorBody: "La connexion n'a pas abouti ; relancez-la.",
    busyHeading: 'Presque terminé',
    busyBody: wait =>
      `Hermes n'a pas pu terminer votre connexion car le service Nous est occupé. Réessayez dans ${wait}. Votre session reste disponible entre-temps.`,
    unreachableBody:
      "Hermes n'a pas pu joindre le service Nous pour terminer votre connexion. Vérifiez votre connexion Internet et réessayez. Votre session reste disponible.",
    alreadySignedInHeading: 'Déjà connecté.',
    alreadySignedInBody: 'Cette installation Hermes est déjà connectée à un compte Nous.',
    setupFailed: {
      gateClosed:
        'Cette version de Hermes ne peut pas démarrer sans compte Nous. Connectez-vous ou créez-en un gratuitement en une minute.',
      paused:
        "L'utilisation de Hermes sans connexion est momentanément suspendue. Hermes continuera à vérifier. La connexion est gratuite et vous permet de continuer immédiatement.",
      rateLimited: wait =>
        `Beaucoup de personnes démarrent en ce moment ; Hermes réessaiera dans ${wait}. La connexion est gratuite et évite l'attente.`,
      unreachable:
        "Hermes n'a pas pu joindre le service Nous. Vérifiez votre connexion Internet, puis appuyez sur Réessayer. Vous pouvez aussi connecter un autre fournisseur.",
      serverError:
        'Le service Nous a rencontré un problème. Réessayez dans un instant ou connectez un autre fournisseur.',
      powRequired:
        "Le serveur Nous a demandé une preuve de travail qui n'est pas encore gérée par votre Agent. Connectez-vous ou créez un compte Nous gratuit pour continuer.",
      locked:
        'Cette session ne peut pas continuer sans connexion. Connectez-vous ou créez un compte Nous gratuit pour poursuivre.',
      generic:
        "Hermes n'a pas pu configurer l'accès gratuit sans connexion. Connectez-vous gratuitement ou choisissez un autre fournisseur.",
      signInBelow: 'La connexion est gratuite. Choisissez Nous ci-dessous.',
      tryAgain: 'Réessayer',
      retrying: 'Nouvelle tentative…'
    }
  },
  modelPicker: {
    title: 'Changer de modèle',
    current: 'actuel :',
    unknown: '(inconnu)',
    search: 'Filtrer les fournisseurs et modèles...',
    noModels: 'Aucun modèle trouvé.',
    addProvider: 'Ajouter un fournisseur',
    loadFailed: 'Impossible de charger les modèles',
    loadingIntoMemory: 'Chargement en mémoire',
    downloading: 'Téléchargement',
    localDownloadsHeading: 'Local',
    noAuthenticatedProviders: 'Aucun fournisseur authentifié.',
    pro: 'Pro',
    proNeedsSubscription: 'Les modèles Pro nécessitent un abonnement payant Nous.',
    free: 'Gratuit',
    freeTier: 'Gratuit',
    priceTitle: 'Prix entrant / sortant par million de jetons',
    wasPrice: 'était',
    customModel: 'Modèle personnalisé',
    addCustomModelAction: 'Ajouter un modèle personnalisé…',
    customModelPlaceholder: 'Saisissez un ID de modèle, p. ex. openai/gpt-5'
  },
  modelVisibility: {
    title: 'Modèles',
    search: 'Rechercher des modèles',
    noAuthenticatedProviders: 'Aucun fournisseur authentifié.',
    addProvider: 'Ajouter un fournisseur…',
    addCustomModel: 'Ajouter un modèle personnalisé',
    removeCustomModel: 'Retirer le modèle personnalisé'
  },
  shell: {
    windowControls: 'Contrôles de fenêtre',
    paneControls: 'Contrôles de panneau',
    appControls: "Contrôles d'application",
    modelMenu: {
      search: 'Rechercher des modèles',
      noModels: 'Aucun modèle trouvé',
      editModels: 'Modifier les modèles…',
      refreshModels: 'Actualiser les modèles',
      fast: 'Rapide'
    },
    modelOptions: {
      noOptions: 'Aucune option pour ce modèle',
      options: 'Options',
      thinking: 'Réflexion',
      fast: 'Rapide',
      effort: 'Effort',
      minimal: 'Minimal',
      low: 'Faible',
      medium: 'Moyen',
      high: 'Élevé',
      xhigh: 'Très élevé',
      max: 'Max',
      ultra: 'Ultra',
      sendsOnRoute: (level: string) => `envoie ${level} sur cette route`,
      updateFailed: "Échec de la mise à jour de l'option du modèle",
      fastFailed: 'Échec de la mise à jour du mode rapide'
    },
    gatewayMenu: {
      gateway: 'Gateway',
      connected: 'Connecté',
      connecting: 'Connexion',
      offline: 'Hors ligne',
      inferenceReady: 'Inférence prête',
      inferenceNotReady: 'Inférence non prête',
      checkingInference: "Vérification de l'inférence",
      disconnected: 'Déconnecté',
      reconnectGateway: 'Reconnecter le gateway',
      openSystem: 'Ouvrir le panneau système',
      connection: label => `Connexion : ${label}`,
      recentActivity: 'Activité récente',
      viewAllLogs: 'Voir tous les journaux →',
      messagingPlatforms: 'Plateformes de messagerie'
    },
    approvalMode: {
      title: "Mode d'approbation",
      ariaLabel: mode => `Mode d'approbation : ${mode}`,
      manual: 'Manuel',
      manualDescription: 'Demander avant les actions nécessitant une approbation',
      smart: 'Intelligent',
      smartDescription: 'Évaluer automatiquement les actions et demander quand nécessaire',
      off: 'Désactivé',
      offDescription: "Exécuter sans invites d'approbation"
    },
    statusbar: {
      unknown: 'inconnu',
      restart: 'redémarrage',
      update: 'mise à jour',
      updateInProgress: 'Mise à jour en cours',
      commitsBehind: (count, branch) => `${count} commit${count === 1 ? '' : 's'} en retard sur ${branch}`,
      desktopVersion: version => `Hermes Desktop v${version}`,
      backendVersion: version => `Backend v${version}`,
      clientLabel: version => `client v${version}`,
      connectionSsh: host => `SSH : ${host}`,
      connectionRemote: host => `Distant : ${host}`,
      connectionCloud: host => `Cloud : ${host}`,
      connectionCloudTooltip: host => `Hermes Cloud · ${host}`,
      connectionSshTooltip: host => `SSH · ${host}`,
      connectionRemoteTooltip: host => `Distant · ${host}`,
      backendLabel: version => `backend v${version}`,
      commit: sha => `commit ${sha}`,
      branch: branch => `branche ${branch}`,
      closeCommandCenter: 'Fermer le centre de commandes',
      openCommandCenter: 'Ouvrir le centre de commandes',
      showTerminal: 'Afficher le terminal',
      hideTerminal: 'Masquer le terminal',
      gateway: 'Gateway',
      gatewayReady: 'prêt',
      gatewayNeedsSetup: 'nécessite une configuration',
      gatewayUnavailable: 'inférence indisponible',
      gatewayChecking: 'vérification',
      gatewayConnecting: 'connexion',
      gatewayOffline: 'hors ligne',
      gatewayRestarting: 'redémarrage…',
      gatewayTitle: 'Gateway',
      customizeTitle: "Afficher dans la barre d'état",
      hideStatusbar: "Masquer la barre d'état",
      resetStatusbar: 'Réinitialiser les valeurs par défaut',
      toggleApprovalMode: 'Approbations',
      toggleBackendVersion: 'Version du backend',
      toggleCacheHitRate: 'Taux de cache',
      toggleCommandCenter: 'Centre de commandes',
      toggleContextUsage: 'Jauge de contexte',
      toggleRunningTimer: 'Minuteur de tour',
      toggleSessionTimer: 'Minuteur de session',
      toggleTerminal: 'Terminal',
      toggleTokensPerSecond: 'Tokens par seconde',
      toggleVersion: 'Version et mises à jour',
      toggleFreeTier: 'Offre gratuite',
      toggleWorkspace: 'Espace de travail',
      cacheHitRateTitle:
        'Taux de cache des prompts pour cette session — les tokens en cache coûtent moins cher, un taux élevé est donc plus économique',
      tokensPerSecondTitle: 'Tokens de sortie par seconde, moyenne calculée sur les 10 derniers appels au modèle',
      agents: 'Agents',
      closeAgents: 'Fermer les agents',
      openAgents: 'Ouvrir les agents',
      subagents: count => `${count} sous-agent${count === 1 ? '' : 's'}`,
      failed: count => `${count} en échec`,
      running: count => `${count} en cours`,
      cron: 'Cron',
      openCron: 'Ouvrir les tâches cron',
      webhooks: 'Webhooks',
      openWebhooks: 'Ouvrir les webhooks',
      starmap: 'Graphique de mémoire',
      openStarmap: 'Ouvrir le graphique de mémoire',
      turnRunning: 'En cours',
      contextUsage: 'Utilisation du contexte',
      systemResources: {
        title: 'Ressources système',
        loading: 'Ressources…',
        gpuUtilization: 'Utilisation du GPU',
        gpuMemory: 'Mémoire GPU',
        ram: 'RAM',
        unifiedNote: 'Mémoire unifiée — le GPU et le système partagent cet espace.',
        toggle: 'Ressources système'
      },
      contextUsagePanel: {
        categories: {
          conversation: 'Conversation',
          mcp: 'MCP',
          memory: 'Mémoire',
          rules: 'Règles',
          skills: 'Skills',
          subagent_definitions: 'Définitions de sous-agents',
          system_prompt: 'Invite système',
          tool_definitions: "Définitions d'outils"
        },
        empty: 'Aucune donnée de contexte pour le moment',
        loading: 'Chargement du détail…',
        percentFull: percent => `${percent}% plein`,
        title: 'Utilisation du contexte',
        tokenSummary: (used, max) => `${used} / ${max} jetons`
      },
      focusedSince: 'Focalisé depuis',
      focusedSinceTitle: 'Temps depuis que cette conversation est au premier plan — pas la durée d’un tour',
      yoloOn: 'YOLO activé — approbation automatique des commandes dangereuses. Shift+clic pour basculer globalement.',
      yoloOff: 'YOLO désactivé. Shift+clic pour basculer globalement.',
      modelNone: 'aucun',
      noModel: 'aucun modèle',
      switchModel: 'Changer de modèle',
      openModelPicker: 'Ouvrir le sélecteur de modèles',
      modelPinned:
        "épinglé par vous ; les nouvelles conversations l'utilisent au lieu de la valeur par défaut des paramètres",
      modelTitle: (provider, model) => `Modèle · ${provider} : ${model}`,
      providerModelTitle: (provider, model) => `${provider} · ${model}`
    }
  },
  rightSidebar: {
    aria: 'Barre latérale droite',
    panelsAria: 'Panneaux de la barre latérale droite',
    files: 'Système de fichiers',
    terminal: 'Terminal',
    noFolderSelected: 'Aucun dossier sélectionné',
    changeCwdTitle: 'Changer le répertoire de travail',
    remotePickerTitle: 'Choisir un dossier distant',
    remotePickerDescription: 'Parcourez les dossiers sur le backend connecté.',
    remotePickerSelect: 'Sélectionner le dossier',
    remotePickerNewFolder: 'Nouveau dossier',
    remotePickerFolderName: 'Nom du dossier',
    remotePickerCreateFolder: 'Créer le dossier',
    remotePickerInvalidFolderName: 'Saisissez un seul nom de dossier, sans barre oblique.',
    remotePickerCreateFolderFailed: error => `Impossible de créer le dossier (${error}).`,
    folderTip: cwd => cwd,
    openFolder: 'Ouvrir le dossier',
    refreshTree: "Actualiser l'arbre",
    collapseAll: 'Réduire tous les dossiers',
    showIgnored: 'Afficher les fichiers ignorés par Git',
    hideIgnored: 'Masquer les fichiers ignorés par Git',
    previewUnavailable: 'Aperçu indisponible',
    couldNotPreview: path => `Impossible d'apercevoir ${path}`,
    noProjectTitle: 'Aucun projet',
    noProjectBody: 'Ouvrez un projet pour parcourir ses fichiers et revoir les modifications.',
    noProjectOpen: 'Aucun projet ouvert',
    noDiffs: 'Aucune différence',
    unreadableTitle: 'Illisible',
    unreadableBody: error => `Impossible de lire ce dossier (${error}).`,
    emptyTitle: 'Vide',
    emptyBody: 'Ce dossier est vide.',
    treeErrorTitle: "Erreur de l'arbre",
    treeErrorBody: "L'arborescence des fichiers a rencontré une erreur lors du rendu de ce dossier.",
    tryAgain: 'Réessayer',
    loadingTree: "Chargement de l'arborescence des fichiers",
    loadingFiles: 'Chargement des fichiers',
    terminalHide: 'Masquer le terminal',
    terminalsAria: 'Terminaux',
    terminalNew: 'Nouveau terminal',
    terminalCloseOthers: 'Fermer les autres',
    terminalCloseAll: 'Tout fermer',
    addToChat: 'Ajouter à la conversation'
  },
  preview: {
    tab: 'Aperçu',
    closePane: "Fermer le panneau d'aperçu",
    loading: "Chargement de l'aperçu",
    unavailable: 'Aperçu indisponible',
    opening: 'Ouverture...',
    hide: 'Masquer',
    openPreview: "Ouvrir l'aperçu",
    openInBrowser: 'Ouvrir dans le navigateur',
    openInExternal: 'Ouvrir dans une application externe',
    popIn: 'Réintégrer',
    popOut: 'Détacher',
    linkHint: "⌘/Ctrl-clic pour le panneau d'aperçu",
    sourceLineTitle: 'Clic pour sélectionner · shift-clic pour étendre · glisser vers le compositeur',
    source: 'SOURCE',
    renderedPreview: 'APERÇU',
    diff: 'DIFF',
    unknownSize: 'taille inconnue',
    binaryTitle: 'Cela ressemble à un fichier binaire',
    binaryBody: label => `L'aperçu de ${label} peut afficher du texte illisible.`,
    largeTitle: 'Ce fichier est volumineux',
    largeBody: (label, size) => `${label} fait ${size}. Hermes n'affichera que les 512 Ko premiers.`,
    previewAnyway: 'Aperçu quand même',
    truncated: 'Affichage des 512 Ko premiers.',
    noInlineTitle: 'Aucun aperçu en ligne',
    noInlineBody: mimeType => `${mimeType || 'Ce type de fichier'} peut tout de même être joint en tant que contexte.`,
    edit: 'Modifier',
    editing: 'Modification',
    unsavedChanges: 'Modifications non enregistrées',
    saveFailed: message => `Impossible d'enregistrer : ${message}`,
    diskChangedTitle: 'Fichier modifié sur le disque',
    diskChangedBody:
      "Ce fichier a changé depuis que vous l'avez ouvert. L'écraser avec votre version, ou abandonner vos modifications et recharger ?",
    overwrite: 'Écraser',
    discardReload: 'Abandonner et recharger',
    console: {
      deselect: "Désélectionner l'entrée",
      select: "Sélectionner l'entrée",
      copyFailed: 'Impossible de copier la sortie de la console',
      copyEntry: 'Copier cette entrée',
      sendEntry: 'Envoyer cette entrée à la conversation',
      messages: count => `${count} messages de console`,
      resize: "Redimensionner la console d'aperçu",
      title: "Console d'aperçu",
      selected: count => `${count} sélectionné(s)`,
      sendToChat: 'Envoyer à la conversation',
      copySelected: 'Copier la sélection dans le presse-papiers',
      copyAll: 'Tout copier dans le presse-papiers',
      copy: 'Copier',
      clear: 'Effacer',
      empty: 'Aucun message de console pour le moment.',
      promptHeader: "Console d'aperçu :",
      sentTitle: 'Envoyé à la conversation',
      sentMessage: count => `Ajout au compositeur : ${count} message${count === 1 ? '' : 's'} de console`
    },
    web: {
      appFailedToBoot: "Échec du démarrage de l'application d'aperçu",
      serverNotFound: 'Serveur non trouvé',
      remoteLoopback:
        "Cette adresse pointe vers la machine qui exécute votre agent, pas vers celle-ci. Le panneau du navigateur charge les pages localement ; un serveur de développement distant nécessite donc une redirection de port ou un nom d'hôte accessible.",
      failedToLoad: "Échec du chargement de l'aperçu",
      tryAgain: 'Réessayer',
      restarting: 'Hermes redémarre...',
      askRestart: 'Demander à Hermes de redémarrer le serveur',
      lookingRestart: taskId => `Hermes recherche un serveur d'aperçu à redémarrer (${taskId})`,
      restartingTitle: "Redémarrage du serveur d'aperçu",
      restartingMessage: "Hermes travaille en arrière-plan. Surveillez la console d'aperçu pour suivre la progression.",
      startRestartFailed: message => `Impossible de démarrer le redémarrage du serveur : ${message}`,
      restartFailed: 'Échec du redémarrage du serveur',
      hideConsole: "Masquer la console d'aperçu",
      showConsole: "Afficher la console d'aperçu",
      hideDevTools: "Masquer les outils de développement d'aperçu",
      openDevTools: "Ouvrir les outils de développement d'aperçu",
      goBack: 'Retour',
      goForward: 'Suivant',
      reload: 'Recharger la page',
      address: 'Adresse',
      addressPlaceholder: 'Saisir une adresse',
      blankPageBody: "Saisissez une adresse ci-dessus pour naviguer, ou demandez à Hermes d'ouvrir une page.",
      finishedRestarting: message =>
        `Hermes a terminé le redémarrage du serveur d'aperçu${message ? `: ${message}` : ''}`,
      failedRestarting: message => `Échec du redémarrage du serveur : ${message}`,
      unknownError: 'erreur inconnue',
      restartedTitle: "Serveur d'aperçu redémarré",
      reloadingNow: "Rechargement de l'aperçu maintenant.",
      restartFailedTitle: "Échec du redémarrage de l'aperçu",
      restartFailedMessage: "Hermes n'a pas pu redémarrer le serveur.",
      stillWorking:
        "Hermes travaille toujours, mais aucun résultat de redémarrage n'est arrivé. La commande du serveur peut être en cours d'exécution au premier plan.",
      workspaceReloading: "Espace de travail modifié, rechargement de l'aperçu",
      fileChanged: url => `Fichier modifié, rechargement de l'aperçu : ${url}`,
      filesChanged: (count, url) => `${count} modifications de fichier, rechargement de l'aperçu : ${url}`,
      watchFailed: message => `Impossible de surveiller le fichier d'aperçu : ${message}`,
      moduleMimeDescription:
        "Les scripts de module sont servis avec le mauvais type MIME. Cela signifie généralement qu'un serveur de fichiers statiques sert une application Vite/React au lieu du serveur de développement du projet.",
      loadFailedConsole: (code, message) => `Échec du chargement${code ? ` (${code})` : ''} : ${message}`,
      unreachableDescription: "La page d'aperçu n'a pas pu être atteinte.",
      openTarget: url => `Ouvrir ${url}`,
      fallbackTitle: 'Aperçu',
      annotate: 'Annoter',
      annotateOn: "Arrêter l'annotation",
      annotateNeedPage: "Ouvrez d'abord une page dans le navigateur intégré.",
      annotateFailed: "Impossible de démarrer le mode d'annotation",
      commenting: 'Ajout de commentaires',
      addComments: count => (count === 1 ? 'Ajouter 1 commentaire' : `Ajouter ${count} commentaires`),
      commentPlaceholder: 'Ajouter un commentaire...',
      commentTitle: n => `Commentaire ${n}`,
      saveComment: 'Enregistrer',
      cancelComment: 'Annuler le commentaire'
    }
  },
  interfaceMode: {
    title: 'Mode d’interface',
    hint: 'Modifie ce qui est affiché, pas ce que Hermes peut faire.',
    sessionNote:
      'Défini par le mode Simple. Une modification ici dure le temps de cette session ; passez en mode Avancé pour la conserver.',
    simple: {
      label: 'Simple',
      description:
        'Pour discuter avec Hermes. Barre latérale et conversation ; pas de panneaux terminal, fichiers ou diff.'
    },
    advanced: {
      label: 'Avancé',
      description: 'Pour les développeurs. Terminal, fichiers, diffs, barre d’état et dispositions, selon vos réglages.'
    }
  },
  zones: {
    showTabStrip: 'Afficher les onglets',
    hideTabStrip: 'Masquer les onglets',
    showStripTab: title => `Afficher ${title}`,
    hideStripTab: title => `Masquer ${title}`,
    lastTabKeptTitle: 'Le dernier onglet reste affiché',
    lastTabKeptBody:
      "Cette zone doit conserver au moins un onglet visible. Affichez d'abord un autre onglet ou repliez toute la barre latérale.",
    toggleStripTab: title => `Afficher ou masquer l'onglet ${title}`,
    minimize: 'Réduire',
    restore: 'Restaurer',
    closeRunningTitle: "Fermer l'onglet en cours ?",
    closeRunningBody:
      "Cette conversation est toujours en cours (ou en attente de votre saisie). Fermer l'onglet la masque — la session conserve sa progression et peut être rouverte depuis la barre latérale.",
    closeRunningConfirm: "Fermer l'onglet",
    reload: 'Recharger',
    closeOthers: 'Fermer les autres',
    closeToRight: 'Fermer à droite',
    closeAll: 'Tout fermer',
    newSessionTab: 'Nouvel onglet de session',
    newTab: 'Nouvel onglet',
    pluginDisabled: pluginId => `Plugin « ${pluginId} » désactivé`,
    pluginDisabledBody: 'Réactivez-le dans Paramètres → Plugins pour faire revenir le panneau.',
    missingPane: paneId => `panneau manquant : ${paneId}`,
    editTitle: 'Mises en page',
    editHint:
      'Choisissez une mise en page ou faites glisser les panneaux entre les zones. Clic droit sur une zone pour la diviser.',
    reset: 'Réinitialiser',
    templates: 'Modèles',
    custom: 'Personnalisé',
    newGridLayout: 'Nouvelle mise en page en grille',
    saveCurrentAs: "Enregistrer l'agencement actuel en tant que modèle",
    nameLayoutPlaceholder: 'Nommez cette mise en page…',
    deletePreset: name => `Supprimer ${name}`,
    zoneEditorTitle: 'Éditeur de zones',
    editorHintPre: 'clic pour diviser · ',
    editorHintPost:
      ' inverse la ligne · glisser sur les zones pour fusionner · glisser les bords partagés pour redimensionner',
    templateColumns: 'Colonnes',
    templateRows: 'Lignes',
    templateGrid: 'Grille',
    templatePriority: 'Priorité',
    zoneTag: index => `zone ${index}`,
    mergeZones: count => `Fusionner ${count} zones`,
    customZoneName: count => `Personnalisé ${count}-zones`,
    layoutNamePlaceholder: fallback => `Nom de la mise en page (${fallback})`,
    saveApply: 'Enregistrer et appliquer',
    notExpressible: "cet agencement s'entrelace (roue) — pas encore exprimable comme des divisions imbriquées",
    zoneCount: count => `${count} zones`,
    tabCount: count => `${count} onglet${count === 1 ? '' : 's'}`
  },
  contextMenu: {
    link: {
      openInApp: 'Ouvrir dans le navigateur intégré',
      openExternal: 'Ouvrir dans le navigateur externe',
      copyUrl: "Copier l'URL",
      copyResolvedUrl: "Copier l'URL résolue"
    },
    image: {
      copyImage: "Copier l'image",
      copyImageAddress: "Copier l'adresse de l'image",
      saveImageAs: "Enregistrer l'image sous…"
    },
    edit: {
      cut: 'Couper',
      paste: 'Coller',
      selectAll: 'Tout sélectionner',
      addToDictionary: 'Ajouter au dictionnaire'
    },
    page: {
      copyPageUrl: "Copier l'URL de la page",
      inspectElement: "Inspecter l'élément"
    }
  },
  assistant: {
    thread: {
      loadingSession: 'Chargement de la session',
      showEarlier: 'Afficher les messages précédents',
      loadingResponse: 'Hermes charge une réponse',
      loadingLocalModel: model => `Chargement de ${model} en mémoire`,
      processingPrompt: "Traitement de l'invite",
      resumeWhenBackgroundDone: count =>
        count === 1
          ? 'Reprendra une fois la tâche en arrière-plan terminée'
          : `Reprendra une fois que ${count} tâches en arrière-plan seront terminées`,
      thinking: 'Réflexion en cours',
      thought: 'A réfléchi',
      thoughtBriefly: 'A réfléchi brièvement',
      thoughtFor: duration => `A réfléchi pendant ${duration}`,
      turnDuration: duration => `Ce tour a duré ${duration}`,
      today: time => `Aujourd'hui, ${time}`,
      yesterday: time => `Hier, ${time}`,
      copy: 'Copier',
      refresh: 'Actualiser',
      moreActions: "Plus d'actions",
      branchNewChat: 'Créer une branche dans une nouvelle conversation',
      react: 'Réagir',
      dismissError: "Ignorer l'erreur",
      errorLayers: {
        auth: "Erreur d'authentification",
        billing: 'Crédits épuisés',
        disk: 'Disque plein',
        endpoint: 'Erreur du point de terminaison personnalisé',
        gateway: 'Erreur du gateway',
        generic: 'Échec du tour',
        provider: 'Erreur du fournisseur',
        runtime: "Erreur d'exécution locale",
        streaming: 'Erreur de connexion au flux'
      },
      errorLayerBodies: {
        auth: "Le service d'IA a refusé votre connexion. Vérifiez les identifiants de ce fournisseur, puis renvoyez votre message.",
        billing:
          "Votre compte n'a plus de crédits chez ce fournisseur. Rechargez-le ou changez de fournisseur, puis réessayez.",
        disk: "Votre disque est plein ; Hermes n'a pas pu enregistrer cette conversation. Libérez de l'espace, puis réessayez.",
        endpoint:
          "Hermes ne parvient pas à joindre votre serveur de modèle personnalisé. Vérifiez qu'il fonctionne, puis renvoyez votre message.",
        gateway:
          'Hermes a rencontré un problème interne au démarrage de cette réponse. Renvoyez votre message ; si cela persiste, envoyez les diagnostics.',
        generic:
          "Une erreur s'est produite pendant la réponse de Hermes. Réessayez ou copiez les détails si cela persiste.",
        provider:
          "Le service d'IA n'a pas pu traiter cette demande. Réessayez dans un instant ou changez de fournisseur.",
        runtime:
          'Hermes a rencontré un problème interne au démarrage de cette réponse. Renvoyez votre message ; si cela persiste, envoyez les diagnostics.',
        streaming: 'La connexion a été interrompue avant la fin de la réponse. Réessayez pour la renvoyer.'
      },
      errorCodes: {
        auth: {
          title: provider => `${provider} a refusé votre connexion`,
          body: provider =>
            `Les identifiants enregistrés pour ${provider} ont été refusés. Corrigez-les dans Paramètres ou changez de fournisseur, puis réessayez.`
        },
        auth_permanent: {
          title: provider => `${provider} a refusé votre connexion`,
          body: provider =>
            `Les identifiants enregistrés pour ${provider} sont invalides ou révoqués. Mettez-les à jour ou changez de fournisseur, puis réessayez.`
        },
        billing: {
          title: 'Crédits épuisés',
          body: provider =>
            `Votre compte ${provider} n'a plus de crédits. Rechargez-le ou changez de fournisseur, puis réessayez.`
        },
        rate_limit: {
          title: "Le service d'IA est occupé",
          body: provider => `${provider} limite actuellement les demandes. Attendez une minute, puis réessayez.`
        },
        upstream_rate_limit: {
          title: "Le service d'IA est occupé",
          body: provider => `${provider} limite actuellement les demandes. Attendez une minute, puis réessayez.`
        },
        overloaded: {
          title: "Le service d'IA est surchargé",
          body: provider =>
            `${provider} rencontre des difficultés. Réessayez dans un instant ou changez de fournisseur.`
        },
        server_error: {
          title: "Le service d'IA a rencontré un problème",
          body: provider =>
            `${provider} a renvoyé une erreur serveur. Réessayez dans un instant ou changez de fournisseur.`
        },
        timeout: {
          title: 'Délai de réponse dépassé',
          body: provider => `${provider} n'a pas répondu à temps. Réessayez pour renvoyer le message.`
        },
        stream_drop: {
          title: 'La réponse a été interrompue',
          body: 'La connexion a été coupée avant la fin de la réponse. Réessayez pour la renvoyer.'
        },
        upstream_blocked: {
          title: 'Un pare-feu a bloqué la requête',
          body: (provider: string) =>
            `Un pare-feu ou un CDN placé devant ${provider} a bloqué la requête avant qu’elle n’atteigne le modèle — votre clé est probablement valide. Définissez un en-tête User-Agent via les extra_headers du fournisseur dans Paramètres, ou changez de fournisseur, puis renvoyez votre message.`
        },
        ssl_cert_verification: {
          title: 'Échec de la connexion sécurisée',
          body: provider =>
            `Hermes n'a pas pu vérifier la connexion sécurisée à ${provider}. Vérifiez votre réseau ou votre proxy, ou changez de fournisseur.`
        },
        context_overflow: {
          title: 'Cette conversation est trop longue',
          body: 'La conversation ne tient plus dans le modèle. Compressez-la ou démarrez-en une nouvelle, puis réessayez.'
        },
        payload_too_large: {
          title: 'Ce message est trop volumineux',
          body: 'La demande était trop grande pour le modèle. Compressez la conversation ou démarrez-en une nouvelle, puis réessayez.'
        },
        model_not_found: {
          title: "Ce modèle n'est pas disponible",
          body: provider =>
            `${provider} ne propose pas ce modèle pour votre compte. Choisissez-en un autre, puis réessayez.`
        },
        provider_policy_blocked: {
          title: 'Ce modèle est bloqué par les paramètres de votre compte',
          body: provider =>
            `${provider} n'a pas routé cette demande avec vos paramètres de données ou de confidentialité. Choisissez un autre modèle ou fournisseur.`
        },
        content_policy_blocked: {
          title: "Le service d'IA a refusé cette demande",
          body: provider => `${provider} n'a pas répondu à ce message. Modifiez-le puis renvoyez-le.`
        },
        format_error: {
          title: "Le service d'IA a rejeté la demande",
          body: provider =>
            `${provider} n'a pas accepté la forme de cette demande. Changez de fournisseur ou envoyez les diagnostics.`
        },
        truncated: {
          title: 'La réponse a été écourtée',
          body: "Le modèle s'est arrêté avant la fin. Réessayez pour obtenir une réponse complète."
        },
        invalid_response: {
          title: "Le service d'IA a envoyé une réponse illisible",
          body: provider => `${provider} a renvoyé une réponse que Hermes n'a pas pu lire. Réessayez dans un instant.`
        },
        empty_response: {
          title: "Le service d'IA a envoyé une réponse vide",
          body: provider => `${provider} n'a rien renvoyé pour ce message. Réessayez dans un instant.`
        },
        loop_error: {
          title: 'Hermes est resté bloqué dans une boucle',
          body: 'La réponse répétait les mêmes étapes ; Hermes l’a donc arrêtée. Réessayez ou démarrez une nouvelle conversation.'
        },
        SESSION_NOT_OWNED: {
          title: 'Cette conversation est ouverte ailleurs',
          body: 'Cette conversation est déjà ouverte dans une autre fenêtre Hermes ou un terminal. Fermez-la là-bas puis réessayez, ou démarrez-en une nouvelle ici.'
        },
        disk_full: {
          title: 'Disque plein',
          body: "Votre disque est plein ; Hermes n'a pas pu enregistrer cette conversation. Libérez de l'espace puis réessayez."
        },
        free_tier_disabled: {
          title: "L'utilisation de Hermes sans connexion est désactivée pour le moment",
          body: 'Connectez-vous avec un compte Nous gratuit pour continuer.'
        },
        free_tier_rate_limited: {
          title: 'Vous avez épuisé le quota sans connexion',
          body: 'Il sera bientôt renouvelé. Connectez-vous avec un compte Nous gratuit pour obtenir un quota plus élevé.'
        },
        free_tier_at_capacity: {
          title: 'Le service sans connexion est très sollicité',
          body: 'Connectez-vous gratuitement pour éviter la file, ou réessayez un peu plus tard.'
        },
        free_tier_model_not_free: {
          title: "Ce modèle n'est pas disponible sans connexion",
          body: 'Hermes utilise le modèle gratuit pour le moment. Connectez-vous avec un compte Nous gratuit pour accéder à plus de modèles.'
        },
        free_tier_route: {
          title: "Hermes n'a pas pu joindre le modèle gratuit par cette route",
          body: 'Connectez-vous avec un compte Nous gratuit ou vérifiez le paramètre NOUS_INFERENCE_BASE_URL.'
        },
        free_tier_outage: {
          title: 'Le modèle gratuit rencontre des difficultés',
          body: 'Renvoyez votre message dans une minute.'
        },
        free_tier_refused: {
          title: "Hermes n'a pas pu envoyer ce message sans connexion",
          body: 'La connexion avec un compte Nous est gratuite.'
        }
      },
      errorAuthKinds: {
        api_key: {
          title: provider => `${provider} a refusé votre clé API`,
          body: provider =>
            `La clé enregistrée pour ${provider} est invalide ou révoquée. Mettez-la à jour, puis réessayez.`
        },
        oauth: {
          title: provider => `Votre connexion à ${provider} a expiré`
        }
      },
      errorDetails: 'Détails',
      errorGenericProvider: "Le service d'IA",
      errorToastTitle: "Hermes n'a pas pu terminer la réponse",
      errorRetry: 'Réessayer',
      errorLimitResets: (time: string) => `Le quota se réinitialise à ${time}`,
      errorRetryAtReset: (time: string) => `Réessayer à la réinitialisation du quota (${time})`,
      errorRetryScheduled: (time: string, wait: string) => `Nouvel essai à ${time} — dans ${wait}`,
      errorRetryScheduledCancel: 'Annuler',
      errorStartNewSession: 'Démarrer une nouvelle session',
      errorSwitchProvider: 'Changer de fournisseur',
      errorChooseModel: 'Choisir un modèle',
      errorCompressConversation: 'Compresser la conversation',
      errorCompressFailed: 'Impossible de compresser la conversation',
      errorOpenHermesFolder: 'Ouvrir le dossier Hermes',
      errorOpenHermesFolderFailed: "Impossible d'ouvrir le dossier Hermes",
      errorUpdateApiKey: 'Mettre à jour la clé API',
      errorSignInAgain: provider => `Se reconnecter à ${provider}`,
      errorSignInFreeTier: 'Se connecter avec un compte Nous',
      errorOauthExpired: provider =>
        `Votre connexion à ${provider} a expiré ou a été révoquée. Reconnectez-vous pour continuer la conversation.`,
      errorOpenLogs: 'Ouvrir les journaux',
      errorOpenLogsFailed: "Impossible d'ouvrir le dossier des journaux",
      errorOpenDesktopLogs: 'Ouvrir les journaux du Desktop',
      errorCopyDiagnostics: "Copier les détails de l'erreur",
      errorSendDiagnostics: 'Envoyer les diagnostics',
      filesChanged: count => (count === 1 ? '1 fichier modifié' : `${count} fichiers modifiés`),
      reviewChanges: 'Examiner',
      readAloudFailed: 'Échec de la lecture à voix haute',
      preparingAudio: "Préparation de l'audio…",
      stopReading: 'Arrêter la lecture',
      readAloud: 'Lire à voix haute',
      editMessage: 'Modifier le message',
      expandMessage: 'Développer le message',
      scrollToBottom: 'Défiler vers le bas',
      stop: 'Arrêter',
      restorePrevious: 'Restaurer le point de contrôle précédent',
      restoreCheckpoint: 'Restaurer le point de contrôle',
      restoreFromHere: 'Restaurer le point de contrôle — relancer à partir de cette invite',
      restoreTitle: 'Restaurer ce point de contrôle ?',
      restoreBody:
        "Tout ce qui suit cette invite est retiré de la conversation, puis l'invite est relancée à partir d'ici.",
      restoreConfirm: 'Restaurer et relancer',
      restoreNext: 'Restaurer le point de contrôle suivant',
      goForward: 'Avancer',
      sendEdited: 'Envoyer le message modifié',
      attachingFile: 'Ajout en cours…'
    },
    approval: {
      gatewayDisconnected: "La Gateway Hermes n'est pas connectée",
      sendFailed: "Impossible d'envoyer la réponse d'approbation",
      reconnect: 'Se reconnecter',
      timedOutSystemLine:
        "Le délai d'approbation a expiré — la commande n'a pas été exécutée. Demandez à Hermes de réessayer ou augmentez la limite dans Paramètres → Sécurité → Délai d'approbation.",
      openSafetySettings: 'Ouvrir les paramètres de sécurité',
      run: 'Exécuter',
      command: 'Commande',
      moreOptions: "Plus d'options d'approbation",
      allowSession: 'Autoriser cette session',
      alwaysAllowMenu: 'Toujours autoriser…',
      jumpToApproval: 'Approbation requise',
      reject: 'Rejeter',
      alwaysTitle: 'Toujours autoriser cette commande ?',
      alwaysDescription: pattern =>
        `Cela ajoute le motif « ${pattern} » à votre liste d'autorisation permanente (~/.hermes/config.yaml). Hermes ne redemandera plus pour ce type de commande — que ce soit dans cette session ou dans une session future.`,
      alwaysAllow: 'Toujours autoriser'
    },
    clarify: {
      notReady: "La demande de clarification n'est pas encore prête",
      gatewayDisconnected: "La Gateway Hermes n'est pas connectée",
      sendFailed: "Impossible d'envoyer la réponse de clarification",
      loadingQuestion: 'Chargement de la question…',
      other: 'Autre (saisissez votre réponse)',
      placeholder: 'Saisissez votre réponse…',
      skip: 'Passer',
      skipped: 'Ignoré',
      continueLabel: 'Continuer',
      confirmAndContinueLabel: 'Confirmer et continuer',
      answeredBadge: 'Répondu',
      questionProgress: (answered, total) => `${answered} réponse${answered === 1 ? '' : 's'} sur ${total}`,
      lateAnswer: (question, choice) => `Re : « ${question} » — ma réponse : ${choice}`,
      lateAnswerTip: 'Rédiger cette réponse comme message de suivi',
      lateAnswerHint:
        "Cette invite n'attend plus de réponse. Choisissez une option pour la rédiger comme message de suivi."
    },
    catalogInstall: {
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
      skill: (name: string) => `skill ${name}`,
      kind: {
        plugin: 'plugin',
        skill: 'skill'
      },
      tier: {
        official: 'officiel',
        community: 'communauté'
      },
      targetProfile: (profile: string) => `S’installe dans votre profil ${profile}`,
      sendFailed: 'Impossible d’envoyer votre réponse. Réessayez.',
      commitLabel: 'Commit',
      subdirLabel: 'Dossier',
      securityHeading: 'Sécurité',
      scan: {
        passed: 'Analyse réussie',
        warnings: 'L’analyse a relevé des avertissements',
        failed: 'Échec de l’analyse'
      },
      requirementsLabel: 'Nécessite',
      credentialsHeading: 'Identifiants'
    },
    mcpSetup: {
      installTitle: 'Ajouter des serveurs MCP',
      enableTitle: 'Activer des serveurs MCP',
      authorizeTitle: 'Autoriser des serveurs MCP',
      installAction: 'Installer',
      enableAction: 'Activer',
      authorizeAction: 'Autoriser',
      installed: server => `${server} installé`,
      enabled: server => `${server} activé`,
      authorized: server => `${server} autorisé`,
      failed: server => `Échec de la configuration de ${server}`,
      toolCount: count => (count === 1 ? '1 outil' : `${count} outils`),
      envRequired: "Renseignez d'abord les identifiants requis",
      sendFailed: "Impossible d'envoyer la réponse de configuration MCP",
      reloadFailed:
        'Serveur enregistré, mais le rechargement des outils MCP a échoué — ils seront chargés à la prochaine session',
      gatewayDisconnected: "La Gateway Hermes n'est pas connectée"
    },
    tool: {
      copyCode: 'Copier le code',
      renderingImage: "Rendu de l'image en cours",
      copyOutput: 'Copier la sortie',
      copyCommand: 'Copier la commande',
      copyContent: 'Copier le contenu',
      copyUrl: "Copier l'URL",
      copyResults: 'Copier les résultats',
      copyQuery: 'Copier la requête',
      copyFile: 'Copier le fichier',
      copyPath: 'Copier le chemin',
      failedCalls: count => `${count} appel${count === 1 ? '' : 's'} d'outil en échec`,
      skillActivity: {
        loading: 'Chargement du skill',
        loaded: 'Skill chargé',
        loadFailed: 'Échec du chargement du skill',
        readingResource: 'Lecture de la ressource du skill',
        readResource: 'Ressource du skill lue',
        resourceFailed: 'Échec de la lecture de la ressource du skill',
        listing: 'Liste des skills',
        listed: 'Skills listés',
        listFailed: 'Échec de la liste des skills',
        unavailable: 'Résultat du skill indisponible'
      },
      outputAlt: "Sortie de l'outil",
      rawResponse: 'Réponse brute',
      copyActivity: "Copier l'activité",
      recoveredOne: 'Récupéré après 1 étape échouée',
      recoveredMany: count => `Récupéré après ${count} étapes échouées`,
      failedOne: '1 étape a échoué',
      failedMany: count => `${count} étapes ont échoué`,
      statusRunning: 'En cours',
      statusError: 'Erreur',
      statusRecovered: 'Récupéré',
      statusDone: 'Terminé',
      resultUnavailable: 'Résultat indisponible',
      resultInterrupted: 'Interrompu',
      memoryWriteNoted: 'Écriture en mémoire enregistrée',
      actions: {
        read: 'Lu',
        reading: 'Lecture en cours',
        opened: 'Ouvert',
        opening: 'Ouverture en cours',
        failedToOpen: "Échec de l'ouverture",
        searched: 'Recherche effectuée',
        searching: 'Recherche en cours',
        ran: 'Exécuté',
        running: 'Exécution en cours',
        ranCode: 'Code exécuté',
        runningCode: 'Script en cours'
      },
      prefixes: {
        browser: 'Navigateur',
        web: 'Web'
      },
      titleTemplates: {
        actionCommand: (action, command) => `${action} ${command}`,
        actionQuoted: (action, value) => `${action} “${value}”`,
        actionTarget: (action, target) => `${action} ${target}`,
        prefixedDone: (prefix, action) => `${prefix} ${action}`,
        runningPrefixedTool: (prefix, action) => `${prefix.toLowerCase()} ${action.toLowerCase()} en cours`,
        runningTool: action => `${action.toLowerCase()} en cours`
      },
      titles: {
        browser_click: {
          done: 'Élément de la page cliqué',
          pending: "Clic sur l'élément de la page en cours",
          pendingAction: 'Clic en cours'
        },
        browser_fill: {
          done: 'Champ de formulaire rempli',
          pending: 'Remplissage du champ de formulaire en cours',
          pendingAction: 'Remplissage en cours'
        },
        browser_navigate: {
          done: 'Page ouverte',
          pending: 'Ouverture de la page en cours',
          pendingAction: 'Ouverture en cours'
        },
        browser_snapshot: {
          done: 'Instantané de la page capturé',
          pending: "Capture de l'instantané de la page en cours",
          pendingAction: 'Capture en cours'
        },
        browser_take_screenshot: {
          done: "Capture d'écran effectuée",
          pending: "Capture d'écran en cours",
          pendingAction: 'Capture en cours'
        },
        browser_type: {
          done: 'Saisie effectuée sur la page',
          pending: 'Saisie sur la page en cours',
          pendingAction: 'Saisie en cours'
        },
        clarify: {
          done: 'Question posée',
          pending: 'Question en cours',
          pendingAction: 'Question'
        },
        cronjob: {
          done: 'Tâche cron',
          pending: 'Planification de la tâche cron en cours',
          pendingAction: 'Planification en cours'
        },
        edit_file: {
          done: 'Fichier modifié',
          pending: 'Modification du fichier en cours',
          pendingAction: 'Modification en cours'
        },
        execute_code: {
          done: 'Code exécuté',
          pending: 'Script en cours',
          pendingAction: 'Script en cours'
        },
        image_generate: {
          done: 'Image générée',
          pending: "Génération de l'image en cours",
          pendingAction: 'Génération en cours'
        },
        list_files: {
          done: 'Fichiers listés',
          pending: 'Liste des fichiers en cours',
          pendingAction: 'Liste en cours'
        },
        memory: {
          done: 'Enregistré dans la mémoire',
          pending: 'Enregistrement dans la mémoire en cours',
          pendingAction: 'Enregistrement en cours'
        },
        patch: {
          done: 'Fichier corrigé',
          pending: 'Correction du fichier en cours',
          pendingAction: 'Correction en cours'
        },
        read_file: {
          done: 'Fichier lu',
          pending: 'Lecture du fichier en cours',
          pendingAction: 'Lecture en cours'
        },
        search_files: {
          done: 'Fichiers recherchés',
          pending: 'Recherche dans les fichiers en cours',
          pendingAction: 'Recherche en cours'
        },
        session_search_recall: {
          done: 'Historique de session recherché',
          pending: "Recherche dans l'historique de session en cours",
          pendingAction: 'Recherche en cours'
        },
        terminal: {
          done: 'Commande exécutée',
          pending: 'Exécution de la commande en cours',
          pendingAction: 'Exécution en cours'
        },
        todo: {
          done: 'Tâches mises à jour',
          pending: 'Mise à jour des tâches en cours',
          pendingAction: 'Mise à jour en cours'
        },
        vision_analyze: {
          done: 'Image analysée',
          pending: "Analyse de l'image en cours",
          pendingAction: 'Analyse en cours'
        },
        web_extract: {
          done: 'Page web lue',
          pending: 'Lecture de la page web en cours',
          pendingAction: 'Lecture en cours'
        },
        web_search: {
          done: 'Recherche web effectuée',
          pending: 'Recherche web en cours',
          pendingAction: 'Recherche en cours'
        },
        write_file: {
          done: 'Fichier modifié',
          pending: 'Modification du fichier en cours',
          pendingAction: 'Modification en cours'
        }
      }
    }
  },
  prompts: {
    gatewayDisconnected: "La Gateway Hermes n'est pas connectée",
    reconnect: 'Se reconnecter',
    sudoSendFailed: "Impossible d'envoyer le mot de passe sudo",
    secretSendFailed: "Impossible d'envoyer le secret",
    sudoTitle: 'Mot de passe administrateur',
    sudoDesc:
      "Hermes a besoin de votre mot de passe sudo pour exécuter une commande privilégiée. Il n'est envoyé qu'à votre agent local.",
    sudoCommandUnavailable:
      "Cet agent n'a pas fourni la commande. Annulez si vous ne pouvez pas la vérifier dans la conversation.",
    sudoInstallDesc:
      'Hermes a besoin de votre mot de passe sudo pour installer les paquets de Bot Screen (TigerVNC + Xfce) sur l’hôte du gateway. Il n’est envoyé qu’à cet hôte.',
    sudoPlaceholder: 'mot de passe sudo',
    secretTitle: 'Secret requis',
    secretDesc: "Hermes a besoin d'un identifiant pour continuer.",
    secretPlaceholder: 'valeur du secret',
    vaultUnlockSendFailed: "Impossible d'envoyer le mot de passe principal",
    vaultUnlockTitle: name => `Déverrouiller ${name}`,
    vaultUnlockDesc: name =>
      `L'agent souhaite se connecter à un site avec un identifiant enregistré dans ${name}. Saisissez votre mot de passe principal pour le déverrouiller pendant cette session : il est envoyé directement à ${name} sur cet ordinateur, sans être enregistré ni montré à l'agent.`,
    vaultUnlockPlaceholder: 'Mot de passe principal',
    vaultUnlockKeepLocked: 'Laisser verrouillé',
    vaultUnlockConfirm: 'Déverrouiller',
    vaultSaveSendFailed: "Impossible d'enregistrer l'identifiant",
    vaultSaveTitle: site => `Enregistrer votre identifiant ${site} ?`,
    vaultSaveDesc: origin =>
      `Hermes a atteint une page de connexion sur ${origin} et ne possède aucun identifiant pour celle-ci. Saisissez-le une fois ici : il sera chiffré sur cet ordinateur et rempli dans la page sans que le modèle voie le mot de passe.`,
    vaultSaveIdentifierLabel: "Adresse e-mail ou nom d'utilisateur",
    vaultSaveIdentifierPlaceholder: 'vous@exemple.fr',
    vaultSavePasswordPlaceholder: 'Mot de passe',
    vaultSaveFootnote: 'Gérez les identifiants enregistrés dans Paramètres → Mots de passe et identifiants.',
    vaultSaveDecline: 'Ne pas enregistrer',
    vaultSaveConfirm: 'Enregistrer et se connecter',
    vaultCodeSendFailed: "Impossible d'envoyer le code",
    vaultCodeTitle: site => `Code de vérification pour ${site}`,
    vaultCodeDesc: site =>
      `${site} demande un code à usage unique reçu par SMS, e-mail ou application d'authentification. Saisissez-le ici : Hermes le remplira dans la page sans que le modèle le voie.`,
    vaultCodeLabel: 'Code',
    vaultCodeFootnote:
      "Astuce : enregistrez la clé d'authentification avec cet identifiant dans Paramètres → Mots de passe et identifiants ; Hermes saisira alors les codes pour vous.",
    vaultCodeSkip: 'Ignorer',
    vaultCodeConfirm: 'Saisir le code'
  },
  desktop: {
    audioReadFailed: "Impossible de lire l'audio enregistré",
    sessionUnavailable: 'Session indisponible',
    createSessionFailed: 'Impossible de créer une nouvelle session',
    promptFailed: "Échec de l'invite",
    staleSessionTitle: 'Conversation obsolète',
    staleSessionBody:
      'Cette fenêtre était en retard sur une autre vue du même chat. Les derniers messages ont été chargés. Renvoyez si vous le souhaitez encore.',
    providerCredentialRequired: "Ajoutez un identifiant de fournisseur avant d'envoyer votre premier message.",
    emptySlashCommand: 'commande slash vide',
    desktopCommands: 'Commandes Desktop',
    skillCommandsAvailable: count => `${count} commandes de skill disponibles.`,
    warningLine: message => `avertissement : ${message}`,
    yoloArmed: 'YOLO activé pour cette conversation',
    yoloOff: 'YOLO désactivé',
    yoloSystem: active => `YOLO ${active ? 'on' : 'off'} pour cette session`,
    yoloTitle: 'YOLO',
    yoloToggleFailed: 'Impossible de basculer YOLO',
    profileStatus: current =>
      `Profil : ${current}. Utilisez /profile <name> ou le sélecteur « Nouvelle session » pour démarrer une conversation dans un autre profil.`,
    unknownProfile: 'Profil inconnu',
    noProfileNamed: (target, available) => `Aucun profil nommé « ${target} ». Disponibles : ${available}`,
    newChatsProfile: name => `Les nouvelles conversations utiliseront le profil ${name}.`,
    setProfileFailed: 'Échec de la définition du profil',
    sttDisabled: 'La reconnaissance vocale est désactivée dans les paramètres.',
    stopFailed: "Échec de l'arrêt",
    regenerateFailed: 'Échec de la régénération',
    editFailed: 'Échec de la modification',
    editTurnUnavailable:
      "Ce tour ne figure plus dans l'historique du serveur (il a peut-être été supprimé lors de la compaction).",
    resumeFailed: 'Échec de la reprise',
    readOnlyTranscriptTitle: 'Ouverte en lecture seule',
    readOnlyTranscriptBody:
      "Aucun backend connecté ne revendique encore cette ancienne conversation ; elle est donc ouverte comme transcription en lecture seule. Son historique est intact, mais l'envoi reste désactivé jusqu'à ce qu'un backend la prenne en charge.",
    readOnlyTranscriptSendBlocked:
      "Cette conversation est ouverte comme transcription en lecture seule — l'envoi est désactivé.",
    resumeStrandedTitle: 'Impossible de charger cette session',
    resumeStrandedBody:
      "La connexion à cette session a échoué et les nouvelles tentatives automatiques ont été abandonnées. Vérifiez que la Gateway est en cours d'exécution, puis réessayez.",
    poolSlotTimeoutBody:
      "Tous les emplacements de backend local sont occupés. Augmentez le nombre de backends de bots maintenus actifs dans Paramètres → Avancé, ou réessayez après l'éviction d'un backend inactif.",
    poolSlotTimeoutOpenSettings: 'Ouvrir les paramètres avancés',
    resumeRetry: 'Réessayer',
    nothingToBranch: 'Aucune branche possible',
    branchNeedsChat: 'Démarrez ou reprenez une conversation avant de créer une branche.',
    sessionBusy: 'Session occupée',
    branchStopCurrent: 'Arrêtez le tour actuel avant de créer une branche pour cette conversation.',
    branchNoText: 'Ce message ne contient pas de texte à partir duquel créer une branche.',
    branchTitle: n => `Brouillon : branche nº${n}`,
    branchFailed: 'Échec de la création de la branche',
    deleteFailed: 'Échec de la suppression',
    archived: 'Archivé',
    archiveFailed: "Échec de l'archivage",
    restored: 'Restauré',
    unarchiveFailed: 'Échec de la restauration',
    cwdChangeFailed: 'Échec du changement de répertoire de travail',
    cwdStagedTitle: "Répertoire de travail en attente d'application",
    cwdStagedMessage: 'Redémarrez le backend desktop pour appliquer les modifications de cwd à cette session active.',
    modelSwitchConfirmBody: 'Ce changement de modèle nécessite une confirmation.',
    modelSwitchConfirmLabel: 'Changer quand même',
    modelSwitchConfirmTitle: model => `Passer à ${model} ?`,
    modelSwitchConfirmTitleFallback: 'Changer de modèle ?',
    modelSwitchFailed: 'Échec du changement de modèle',
    modelSwitchKeepLabel: 'Conserver le modèle actuel',
    modelSwitchStaleNotice: "La sélection a changé — le changement de modèle n'a pas été appliqué.",
    hydrationSyncing: (profile: string) => `Synchronisation de ${profile}…`,
    sessionExported: 'Session exportée',
    sessionExportFailed: "Impossible d'exporter la session",
    imageSaved: 'Image enregistrée',
    downloadStarted: 'Téléchargement démarré',
    restartToUseSaveImage: "Redémarrez Hermes Desktop pour utiliser Enregistrer l'image.",
    restartToSaveImages: 'Redémarrez Hermes Desktop pour enregistrer les images',
    imageDownloadFailed: "Échec du téléchargement de l'image",
    openImage: "Ouvrir l'image",
    downloadImage: "Télécharger l'image",
    savingImage: "Enregistrement de l'image en cours",
    imagePreviewFailed: "Échec de l'aperçu de l'image",
    imageAttach: 'Image jointe',
    imageWriteFailed: "Échec de l'écriture de l'image sur le disque.",
    imageAttachFailed: "Échec de l'ajout de l'image",
    pastedContent: 'Contenu collé',
    pasteAttachFailed: 'Impossible de joindre le texte collé',
    attachImages: 'Joindre des images',
    clipboard: 'Presse-papiers',
    noClipboardImage: 'Aucune image trouvée dans le presse-papiers',
    clipboardPasteFailed: 'Échec du collage depuis le presse-papiers',
    dropFiles: 'Déposer des fichiers',
    handoff: {
      pickPlatform: 'Choisissez une destination',
      success: platform => `Transféré vers ${platform}. Reprenez ici à tout moment.`,
      systemNote: platform => `↻ Transféré vers ${platform} — reprenez ici à tout moment.`,
      failed: error => `Échec du transfert : ${error}`,
      timedOut: "Délai d'expiration en attendant la Gateway. `hermes gateway` est-il en cours d'exécution ?",
      startMessaging: 'Démarrer la messagerie'
    }
  },
  tips: {
    close: 'Ne plus afficher cette astuce',
    items: {
      'new-session': {
        title: 'Repartir de zéro',
        text: 'Une nouvelle conversation dispose de son propre contexte, terminal et dossier de travail.'
      },
      skills: {
        title: 'Apprenez-lui une seule fois',
        text: "Les compétences sont des dossiers d'instructions que Hermes charge lorsque le travail le nécessite."
      },
      messaging: {
        title: 'Hermes loin de votre bureau',
        text: 'Connectez Telegram, Discord, Slack et plus encore : même agent, même mémoire.'
      },
      artifacts: {
        title: 'Tout ce que Hermes a créé',
        text: 'Images, fichiers et liens de chaque session, indexés au même endroit.'
      },
      cron: {
        title: "Du travail qui s'exécute tout seul",
        text: 'Planifiez une invite toutes les heures, chaque nuit ou avec une expression cron.'
      },
      'command-palette': {
        title: 'Une seule zone pour tout faire',
        text: 'Sessions, paramètres, compétences et commandes sont accessibles depuis la palette.'
      },
      profiles: {
        title: 'Les profils sont séparés',
        text: 'Chacun possède son propre Hermes, avec ses clés, sa mémoire et ses sessions.'
      },
      'composer-mentions': {
        title: 'Joindre et commander',
        text: 'Saisissez @ pour joindre un fichier à la conversation, ou / pour exécuter une commande.'
      },
      'local-runtime-update': {
        title: 'Une mise à jour du moteur local est disponible',
        text: 'Mettez à jour le moteur qui exécute vos modèles locaux. Les requêtes locales actives peuvent être interrompues.',
        action: 'Mettre à jour maintenant'
      },
      'local-setup': {
        title: 'Cette machine peut exécuter des modèles en local',
        text: 'Votre matériel peut servir un modèle local. Les conversations restent sur votre ordinateur et ne coûtent rien.',
        action: 'Configurer'
      },
      'right-pane': {
        title: 'Le volet de travail',
        text: "Les fichiers, le terminal, la revue et le navigateur intégré partagent le côté droit de l'application."
      }
    }
  },
  errors: {
    genericFailure: "Une erreur s'est produite",
    boundaryTitle: "Un problème est survenu dans l'interface",
    boundaryDesc: 'La vue a rencontré une erreur inattendue. Vos conversations et vos paramètres sont en sécurité.',
    boundaryDetails: 'Détails',
    sendDiagnostics: 'Envoyer les diagnostics',
    reloadWindow: 'Recharger la fenêtre',
    openLogs: 'Ouvrir les journaux'
  },
  ui: {
    search: {
      clear: 'Effacer la recherche'
    },
    pagination: {
      label: 'pagination',
      previous: 'Préc.',
      previousAria: 'Aller à la page précédente',
      next: 'Suiv.',
      nextAria: 'Aller à la page suivante'
    },
    sidebar: {
      title: 'Barre latérale',
      description: 'Affiche la barre latérale mobile.',
      toggle: open => `${open ? 'Afficher' : 'Masquer'} la barre latérale`
    }
  }
} satisfies TranslationOverrides

export const fr = defineLocale(frOverrides)
