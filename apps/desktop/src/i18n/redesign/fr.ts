import type { RedesignOverlay } from './types'

// French wording for the redesigned shell: only the keys whose text differs from the base catalog.
export const frRedesign: RedesignOverlay = {
  sharedMetrics: {
    sending:
      'Si vous partagez, Hermes envoie les statistiques à Nous une fois par jour avec un identifiant aléatoire propre à cet agent. Hormis une note unique indiquant que Hermes a été installé, les statistiques d’avant votre accord ne sont jamais envoyées. Modifiez ce choix à tout moment dans les Paramètres.',
    sendDesc:
      'Uniquement des comptages, envoyés une fois par jour avec un identifiant aléatoire pour cet agent. Jamais vos messages, fichiers, chemins ni textes d’erreur.',
    consentBody:
      'Hermes peut compter la façon dont vous l’utilisez : durée des discussions, modèles et outils utilisés, et moments où quelque chose échoue. Il n’enregistre jamais vos messages, fichiers, chemins ni textes d’erreur.',
    collectedActivity: 'Discussions : durée, résultat, type d’erreur, temps actif par jour',
    collectedMilestones:
      'Configuration : étapes terminées, connexions aux fournisseurs, nombre de skills, plugins et routines'
  },
  firstRun: {
    intro: {
      greeting: 'Bonjour, je suis votre agent.'
    },
    guide: {
      upgradeRequired:
        'Hermes sur cette machine doit être mis à jour pour la configuration sans discussion. Sur cette machine, exécutez hermes update, redémarrez Hermes, puis réessayez. Vos discussions restent inchangées.'
    }
  },
  connectors: {
    disclaimer: 'La connexion est facultative. N’autorisez que les applications que vous voulez confier à l’agent.',
    namedAccount: (app: string, alias: string) => `L’agent a nommé ce compte ${app} ${alias}.`,
    unavailable: 'Les connecteurs ne sont pas disponibles pour cette discussion.'
  },
  connectorsPage: {
    page: {
      noMatchBody: 'Aucun résultat. Indiquez à l’agent votre propre serveur MCP pour l’ajouter.'
    },
    dialog: {
      disconnectBody:
        'Ce compte sera déconnecté pour tous les agents utilisant le même compte Nous, pas seulement pour cet agent. Vous pourrez le reconnecter à tout moment.',
      appSwitch: (name: string) => `L’agent peut utiliser ${name}`,
      bothOn: (name: string) => `Les deux sont activés : l’agent voit donc chaque outil ${name} en double.`,
      nousLine: 'Les applications Nous suivent votre compte, pas l’agent.',
      rulesSignIn: 'Connectez-vous pour modifier ce que l’agent peut faire ici.'
    },
    accounts: {
      removeBody:
        'Ce compte connecté sera retiré pour tous les agents utilisant le même compte Nous, pas seulement pour cet agent. Vos autres comptes restent connectés.'
    },
    tools: {
      summaryTitle: (name: string) => `Ce que l’agent peut faire avec ${name}`,
      summaryPreviewTitle: (name: string) => `Ce que l’agent pourra faire avec ${name} une fois connecté`,
      goneBody: 'L’agent ne peut plus l’appeler. La ligne reste jusqu’à ce que vous la supprimiez : rien ne disparaît.'
    }
  },
  boot: {
    steps: {
      connectingGateway: 'Connexion à la machine',
      loadingSessions: 'Chargement des discussions récentes'
    },
    errors: {
      gatewayConnectionLost: 'Connexion à l’ordinateur perdue',
      gatewayConnectionLostDetail:
        'Nouvelle tentative en arrière-plan. Vous pouvez continuer à lire et rédiger — ouvrez les paramètres de l’ordinateur si le problème persiste.',
      gatewaySignInRequired: 'Authentification sur l’ordinateur requise'
    },
    failure: {
      description:
        "Le service en arrière-plan de Hermes n'a pas pu se lancer. Essayez l'une des étapes de récupération ci-dessous. Rien ici ne supprime vos conversations ou paramètres.",
      remoteTitle: 'Authentification sur la machine distante requise',
      remoteDescription:
        'Votre session sur la machine distante a expiré. Connectez-vous à nouveau pour vous reconnecter. Rien ici ne supprime vos discussions ou paramètres.',
      useLocalGateway: 'Utiliser la machine locale',
      gatewaySettings: 'Paramètres de la machine',
      remoteSignInHint: signInLabel =>
        `Déconnecte la session navigateur distante enregistrée, puis ouvre ${signInLabel}. Utilisez la machine locale pour passer au backend intégré.`,
      remoteFailureHint:
        'Vérifiez l’URL et l’authentification dans les paramètres de la machine, ou passez à la machine locale.',
      cloudDownDescription:
        'L’agent cloud géré par Nous auquel cette application se connecte renvoie une erreur serveur. Il ne peut pas être redémarré depuis ici — vérifiez son état, passez à la machine locale ou contactez l’assistance.',
      signedInMessage: 'Reconnexion à la machine distante…',
      signInToRemoteGateway: 'Se connecter à la machine distante'
    }
  },
  notifications: {
    errors: {
      gatewayAuthFailed:
        'Cette machine n’accepte plus votre connexion enregistrée. Ouvrez Machines et reconnectez-vous (ou collez un nouveau jeton d’accès), puis réessayez.'
    },
    actions: {
      openGateways: 'Ouvrir Machines'
    },
    voice: {
      liveDelegationFailed: 'Impossible de transmettre la demande à l’agent',
      couldNotStartSession: 'Impossible de démarrer la discussion vocale',
      liveEnded: 'Discussion vocale en direct terminée',
      liveEndedConnectionLost: 'La discussion vocale en direct a perdu sa connexion.',
      liveEndedClosed: 'La discussion vocale en direct a été fermée par le service.'
    },
    native: {
      inputBody: 'L’agent attend votre réponse.',
      turnDoneTitle: 'L’agent a terminé'
    }
  },
  promptNotices: {
    sessionCommands: {
      busy: (command: string) =>
        `discussion occupée — l’agent est encore en train de répondre. Arrêtez d’abord la réponse en cours (bouton Arrêter ou Ctrl+C dans un terminal), puis lancez /${command}.`,
      busyCompressing: (command: string) =>
        `discussion occupée — un /compress est en cours ; lancez /${command} quand il sera terminé.`,
      titleSet: (title, queued) =>
        `Titre de la discussion défini : ${title}${queued ? ' (en attente du démarrage de la discussion)' : ''}`,
      titleCleared: 'Titre de la discussion effacé.'
    },
    desktopCommands: {
      configFooter:
        'Modifiez les réglages dans Réglages, ou éditez config.yaml avec `hermes config edit` sur la machine qui exécute Hermes.',
      saveLegacyRefused:
        'Non enregistré : Hermes sur cette machine ne peut pas exporter de copie expurgée, nommée ou autre que JSON. Mettez à jour Hermes, ou lancez /save sans option pour écrire le JSON brut sur cette machine.',
      savedOnHost: (path: string) => `Enregistré sur la machine qui exécute Hermes : ${path}`,
      exportRemote:
        '/export empaquette un agent depuis la machine qui exécute Hermes, qui n’est pas cet appareil. Lancez `hermes profile export` sur cette machine.',
      importRemote:
        '/import installe un agent sur la machine qui exécute Hermes, qui n’est pas cet appareil. Lancez `hermes profile import <archive>` sur cette machine.',
      checkpointsDisabled:
        'Les points de restauration ne sont pas activés pour cette discussion (définissez checkpoints.enabled dans config.yaml).'
    }
  },
  keepAwakeOffer: {
    title: 'Votre ordinateur s’est mis en veille pendant que l’agent travaillait'
  },
  sendDiagnostics: {
    privacyNotice:
      "Cela téléverse un paquet de débogage vers un stockage interne de Nous, et non vers un service de partage public. Il contient des informations système (système d'exploitation, versions, fournisseur et clés API configurées — jamais les clés elles-mêmes) ainsi que les journaux complets de l'agent, du service Hermes et du Desktop (jusqu'à 512 Ko chacun), susceptibles de contenir des conversations, des résultats d'outils et des chemins de fichiers. Les secrets sont expurgés avant l'envoi. Seuls le personnel de Nous et les modérateurs Discord autorisés peuvent consulter le paquet, qui est automatiquement supprimé après 14 jours."
  },
  keybinds: {
    categories: {
      profiles: 'Agents'
    },
    actions: {
      'nav.profiles': 'Ouvrir les agents',
      'nav.cron': 'Ouvrir les routines',
      'view.toggleProfileRail': 'Afficher ou masquer la barre des agents',
      'profile.default': 'Basculer vers l’agent par défaut',
      'profile.switch.1': 'Basculer vers l’agent 1',
      'profile.switch.2': 'Basculer vers l’agent 2',
      'profile.switch.3': 'Basculer vers l’agent 3',
      'profile.switch.4': 'Basculer vers l’agent 4',
      'profile.switch.5': 'Basculer vers l’agent 5',
      'profile.switch.6': 'Basculer vers l’agent 6',
      'profile.switch.7': 'Basculer vers l’agent 7',
      'profile.switch.8': 'Basculer vers l’agent 8',
      'profile.switch.9': 'Basculer vers l’agent 9',
      'profile.switch.10': 'Basculer vers l’agent 10',
      'profile.switch.11': 'Basculer vers l’agent 11',
      'profile.switch.12': 'Basculer vers l’agent 12',
      'profile.switch.13': 'Basculer vers l’agent 13',
      'profile.switch.14': 'Basculer vers l’agent 14',
      'profile.switch.15': 'Basculer vers l’agent 15',
      'profile.switch.16': 'Basculer vers l’agent 16',
      'profile.switch.17': 'Basculer vers l’agent 17',
      'profile.switch.18': 'Basculer vers l’agent 18',
      'profile.next': 'Agent suivant',
      'profile.prev': 'Agent précédent',
      'profile.toggleAll': 'Basculer la vue tous agents',
      'profile.create': 'Créer un agent',
      'agents.new': 'Nouvel agent (page Agents)',
      'nav.artifacts': 'Ouvrir la bibliothèque',
      'session.new': 'Nouvelle discussion',
      'session.newTab': 'Nouvel onglet de discussion',
      'session.next': 'Discussion suivante',
      'session.prev': 'Discussion précédente',
      'session.slot.1': 'Basculer vers la discussion récente 1',
      'session.slot.2': 'Basculer vers la discussion récente 2',
      'session.slot.3': 'Basculer vers la discussion récente 3',
      'session.slot.4': 'Basculer vers la discussion récente 4',
      'session.slot.5': 'Basculer vers la discussion récente 5',
      'session.slot.6': 'Basculer vers la discussion récente 6',
      'session.slot.7': 'Basculer vers la discussion récente 7',
      'session.slot.8': 'Basculer vers la discussion récente 8',
      'session.slot.9': 'Basculer vers la discussion récente 9',
      'session.focusSearch': 'Rechercher des discussions',
      'session.togglePin': 'Épingler / désépingler la discussion actuelle',
      'session.archive': 'Archiver la discussion actuelle',
      'view.toggleSidebar': 'Basculer la barre latérale des discussions',
      'view.cycleSidebarGrouping': 'Changer le regroupement des discussions'
    }
  },
  settings: {
    subpages: {
      gatewayConnection: 'Machine principale'
    },
    nav: {
      gateway: 'Machines'
    },
    plugins: {
      installModal: {
        profileLabel: 'Installer pour l’agent',
        desktopTargetFromPackage:
          'Chargé dans cette application depuis le paquet ci-dessus — identique pour tous les agents'
      }
    },
    vault: {
      otpHint:
        'La « clé d’installation » que le site affiche lorsque vous activez 2FA. Une fois enregistrée, l’agent génère lui-même les codes.'
    },
    notifications: {
      kinds: {
        input: {
          description: 'L’agent a posé une question ou a besoin d’un mot de passe ou d’un secret.'
        }
      }
    },
    searchPlaceholder: {
      gateway: 'Connexion à une machine...',
      sessions: 'Rechercher des discussions archivées...'
    },
    appearance: {
      reactionsDesc: 'Réactions emoji façon iMessage — réagissez aux messages, et l’agent peut réagir aux vôtres.',
      tipsDesc:
        'Une petite bulle désigne occasionnellement une partie de l’application lorsque vous êtes inactif ou lorsque l’agent peut vous aider. Fermer une astuce la masque définitivement.',
      toursDesc:
        'Laissez l’agent vous guider dans l’application en assombrissant l’écran et en mettant chaque étape en évidence.',
      themeProfileNote: profile => `Enregistré pour l’agent ${profile} — chaque agent conserve son propre thème.`,
      pet: {
        intro:
          'Adoptez une mascotte petdex animée qui flotte au-dessus de l’application et réagit aux actions de l’agent — court pendant l’exécution des outils, fête les réussites, boude les erreurs.'
      },
      sessionDensityTitle: 'Densité de la liste des discussions',
      sessionDensityDesc:
        'Choisissez la quantité de contexte affichée sous les titres de discussion dans la barre latérale.'
    },
    uninstallSection: {
      options: {
        full: {
          description:
            'Supprime l’application, l’agent et toutes les données utilisateur : configuration, discussions, routines, secrets, journaux.'
        }
      }
    },
    memoryLayers: {
      noProviderDesc:
        'L’agent utilise uniquement la couche intégrée ci-dessous. Choisissez un fournisseur externe dans Réglages › Mémoire et contexte pour en ajouter un.',
      builtinDesc:
        'MEMORY.md et USER.md sont la mémoire propre de l’agent, indépendante de tout fournisseur externe. Les deux sont chargés dans les discussions par défaut ; les interrupteurs mémoire et profil utilisateur dans Réglages › Mémoire et contexte désactivent chacun. Un fournisseur s’ajoute à cette couche ; il ne la remplace jamais.',
      selectedDesc: current =>
        `Pas encore actif : les discussions continuent d’utiliser ${current} jusqu’à l’enregistrement de ce choix.`,
      providerDesc: name => `${name} ajoute son propre rappel à chaque discussion, en plus de la couche intégrée.`
    },
    about: {
      channel: {
        heldBody: (v: string) =>
          `Cette version est plus récente que la version stable ${v}, et vos discussions utilisent déjà son environnement d’exécution Hermes, que ${v} ne peut pas ouvrir. Hermes reste sur cette version jusqu’à la prochaine version stable.`
      }
    },
    connections: {
      intro:
        'Gérez cet appareil et chaque machine exécutant Hermes qu’il peut atteindre par des connexions distantes, SSH ou Cloud.',
      stagedNote:
        'Chaque machine reste connectée. Les agents, discussions, messageries et routines restent sur leur machine, et Discussions peut toutes les afficher à la fois.',
      launchModeTitle: 'Au démarrage, revenir aux discussions de la dernière machine utilisée',
      launchModeDesc: 'Si cette option est désactivée, les discussions s’ouvrent sur la machine principale.',
      searchPlaceholder: 'Rechercher des machines…',
      noSearchResults: 'Aucune machine ne correspond à votre recherche.',
      kindRemote: 'Machine distante',
      kindRemoteDesc: 'Une machine exécutant Hermes, accessible en HTTP(S) — par le LAN, Tailscale ou Internet.',
      urlTitle: 'URL de la machine',
      headersTitle: 'En-têtes de connexion supplémentaires',
      headersDesc:
        'Envoyés avec chaque requête HTTP et WebSocket vers cette machine, notamment pour les proxys d’accès comme Cloudflare Access (CF-Access-Client-Id / CF-Access-Client-Secret). Les valeurs sont stockées chiffrées. Les en-têtes gérés par Hermes (Authorization, Cookie, Host…) sont ignorés.',
      duplicateUrl: label => `Une connexion à cette URL de machine existe déjà (« ${label} »).`,
      title: 'Machines enregistrées'
    },
    managedUpdates: {
      intro:
        'Mettez à jour de manière transactionnelle les installations SSH gérées par le Desktop : les discussions actives sont laissées se terminer, le dépôt distant est mis à jour, puis chaque agent est restauré avec un reçu corrélé.',
      progress: 'Fin des discussions actives, mise à jour de l’installation distante et restauration des agents…',
      scopesRestored: (profiles: string) => `Agents restaurés : ${profiles}`,
      scopeNotRestored: (profile: string, error: string) => `Agent « ${profile} » non restauré : ${error}`
    },
    gateway: {
      loading: 'Chargement des paramètres de la machine...',
      unavailableTitle: 'Paramètres de la machine indisponibles',
      unavailableDesc: "Le pont IPC du desktop n'expose pas les paramètres de l’ordinateur.",
      webappHostDesc:
        'La Webapp s’exécute toujours sur la machine qui la sert. Les connexions enregistrées ci-dessous sont la liste de cette machine, partagée avec l’app Hermes Desktop ; les autres machines sont accessibles via celle-ci.',
      webappOauthPaste: {
        description:
          'Si l’onglet de connexion s’est arrêté sur une adresse qui ne charge pas (ce navigateur n’est pas sur la machine qui exécute Hermes), collez cette adresse ici.'
      },
      webAccess: {
        pendingOn: "Enregistré. Hermes servira l'application web après son redémarrage.",
        pendingOff: "Désactivé, mais Hermes continue de servir l'application web jusqu'à son redémarrage.",
        unavailableDetail: reason => `Hermes n'a pas pu servir l'application web : ${reason}`,
        restartBusy:
          'Enregistré. Une discussion est en cours, Hermes n’a donc pas redémarré. Choisissez « Redémarrer Hermes » une fois qu’elle est terminée.',
        restartFailed:
          "Enregistré, mais Hermes n'a pas pu redémarrer. Choisissez « Redémarrer Hermes » pour réessayer.",
        restartFailedBecause: reason => `Hermes n'a pas pu redémarrer : ${reason}`,
        startTimedOut: "Hermes n'a pas servi l'application web en 90 secondes.",
        stopTimedOut: "Hermes n'a pas cessé de servir l'application web en 90 secondes.",
        startStages: {
          restarting: "Redémarrage de Hermes pour servir l'application web."
        },
        restartGateway: 'Redémarrer Hermes'
      },
      title: 'Connexion à une machine',
      intro:
        'Local par défaut. Utilisez distant quand cette application doit exécuter Hermes ailleurs. Connectez une machine pour découvrir ses agents.',
      localTitle: 'Machine locale',
      remoteTitle: 'Machine distante',
      remoteAuthHint:
        'Les machines hébergées utilisent OAuth ou un nom d’utilisateur et un mot de passe ; les machines auto-hébergées peuvent utiliser un jeton de session.',
      cloudSavedTitle: 'Machines Cloud enregistrées',
      cloudSavedDesc:
        'Utilisez une machine enregistrée sans modifier celle par défaut. Connectez-vous ci-dessous pour ajouter des instances. Gérez leurs noms et leur connexion dans la liste des connexions enregistrées.',
      cloudUseSaved: 'Utiliser la machine',
      probing: 'Vérification de la méthode d’authentification de cette machine…',
      probeError:
        "Impossible d'atteindre cet ordinateur pour le moment. Vérifiez l'URL — la méthode d'authentification apparaîtra une fois qu'il répondra.",
      authSignedInPassword:
        'Cette machine utilise un nom d’utilisateur et un mot de passe. Vous êtes connecté ; la session se rafraîchit automatiquement.',
      authSignedInOauth: 'Cette machine utilise OAuth. Vous êtes connecté ; la session se rafraîchit automatiquement.',
      authNeedsPassword:
        'Cette machine utilise un nom d’utilisateur et un mot de passe. Connectez-vous pour autoriser cette application desktop.',
      authNeedsOauth: provider =>
        `Cette machine utilise OAuth. Connectez-vous avec ${provider} pour autoriser cette application desktop.`,
      plainTextConfirmTitle: 'Stocker le jeton de connexion en clair ?',
      keychainEncryptionDesc:
        "Désactivé par défaut. Une fois activé, les jetons de connexion et les identifiants de connexion sont chiffrés avec le trousseau du système (Trousseaux d'accès, GNOME Keyring ou Windows DPAPI) ; le système peut demander une autorisation ou un mot de passe. Sinon, ils sont stockés dans des fichiers ordinaires lisibles uniquement par votre compte utilisateur.",
      diagnosticsDesc: 'Afficher desktop.log dans votre gestionnaire de fichiers — utile quand Hermes ne démarre pas.',
      incompleteTitle: 'Configuration de la machine distante incomplète',
      restartingTitle: 'Reconnexion à la machine',
      savedTitle: 'Paramètres de la machine enregistrés',
      reachableTitle: 'Machine distante accessible',
      signedOutMessage: 'Session de la machine distante effacée.',
      failedLoad: 'Échec du chargement des paramètres de la machine',
      testFailed: 'Échec du test de connexion à la machine distante',
      applyFailed: 'Impossible d’appliquer les paramètres de la machine',
      saveFailed: 'Impossible d’enregistrer les paramètres de la machine',
      webappHostTitle: 'Machine Hermes',
      envOverrideTitle: 'Cette connexion a été imposée par la façon dont Hermes a été lancé.'
    },
    profileScope: {
      editsProfile: profile => `Les modifications de cette page s’appliquent à l’agent « ${profile} ».`
    },
    model: {
      tasks: {
        profile_describer: {
          label: 'Description d’agent',
          hint: 'Descriptions automatiques des agents'
        },
        title_generation: {
          hint: 'Titres de discussion'
        }
      },
      subagents: {
        routing: {
          modes: {
            profile_routes: {
              label: 'Routes d’agent',
              description:
                'Les sous-agents utilisent l’un de vos agents, avec son propre modèle, ses skills et ses clés.'
            }
          },
          pluginMissing: 'Routes de modèle et Routes d’agent proviennent du plugin Advanced Delegation.',
          pluginDisabled: 'Advanced Delegation est installé mais désactivé pour cet agent.',
          blockedDisabled:
            'Les sous-agents ne démarreront pas tant qu’Advanced Delegation n’est pas activé pour cet agent.',
          profileRoutesTitle: 'Agents disponibles pour les sous-agents',
          profileRoutesHint: 'Chaque agent coché apporte son propre modèle, ses skills, ses instructions et ses clés.',
          profileEmpty: 'Aucun autre agent pour l’instant. Créez-en un depuis la barre des agents.',
          profileMissing: name => `${name} (l’agent n’existe plus)`,
          inactiveProfiles: profile =>
            `Les routes d’agent choisissent l’agent utilisé par chaque sous-agent. Les sous-agents sans choix explicite utilisent « ${profile} ».`,
          inactiveProfilesNoDefault:
            'Les routes d’agent choisissent l’agent utilisé par chaque sous-agent. Sans choix explicite, les sous-agents utilisent le modèle de l’agent principal.'
        }
      },
      appliesDesc:
        'S’applique aux nouvelles discussions. Utilisez le sélecteur de modèle dans le compositeur pour changer à chaud la discussion active.',
      mainAppliedMessage: model => `Les nouvelles discussions utiliseront ${model}.`
    },
    billing: {
      errors: {
        remoteSpendingReconnect: (who: string) =>
          `${who} Reconnectez-vous depuis Paramètres -> Machines pour réautoriser cet appareil.`,
        sessionRevoked: {
          message: 'Votre session a été déconnectée. Reconnectez-vous depuis Paramètres → Machines.'
        },
        transport: {
          message: 'La requête de facturation a échoué avant d’atteindre Hermes.'
        }
      }
    },
    toolsets: {
      terminalBackend: {
        unavailableMessage: backend =>
          `L’agent ne peut pas exécuter de commandes shell pour le moment : ${backend} n’est pas prêt. Passez en Local ou terminez la configuration de ${backend}, puis réessayez.`,
        needsSetupConfirmDescriptionGeneric:
          'Ce backend n’est pas encore configuré. Les discussions démarrées après ce changement n’auront ni outils de terminal ni outils de fichiers tant que la configuration n’est pas terminée.',
        switchedToLocal:
          'Les commandes de terminal s’exécutent maintenant localement. S’applique aux nouvelles discussions.',
        selectedMessage: backend =>
          `Les commandes de terminal s’exécutent maintenant via ${backend}. S’applique aux nouvelles discussions.`,
        needsSetupConfirmDescription: (detail: string) =>
          `${detail} Les discussions démarrées après ce changement n’auront ni outils de terminal ni outils de fichiers tant que la configuration n’est pas terminée.`
      },
      browserRealProfile: {
        prompt: {
          body: 'Autorisez l’agent à naviguer avec un instantané de votre profil de navigateur par défaut afin que les sites s’ouvrent avec vos sessions déjà connectées, pour l’agent et dans le navigateur intégré.'
        },
        description:
          'Copie les identifiants de connexion et les cookies de votre navigateur par défaut dans un instantané géré que l’agent utilise pour naviguer, et connecte aussi le navigateur intégré avec eux. Votre profil actif n’est jamais ouvert directement. S’applique aux nouvelles discussions.',
        enabledMessage: 'Les nouvelles discussions utiliseront un instantané de votre profil de navigateur par défaut.',
        disabledMessage:
          'L’instantané du profil sera supprimé et le navigateur intégré en sera déconnecté ; les nouvelles discussions utiliseront un navigateur vierge.'
      },
      modelSelectedMessage: model => `${model} s’applique aux nouvelles discussions.`
    },
    fieldDescriptions: {
      'agent.maxTurns': 'Limite supérieure de tours d’appel d’outils avant que l’agent n’arrête une exécution.',
      fileReadMaxChars: 'Nombre maximal de caractères que l’agent peut lire dans une demande de fichier.',
      'approvals.mode': 'Comment l’agent gère les commandes nécessitant une approbation explicite.',
      'voice.voiceChatMode':
        'chained : reconnaissance vocale → agent → synthèse vocale avec les fournisseurs ci-dessous. gpt-live : un modèle vocal OpenAI full-duplex (gpt-live-1) écoute et parle, et confie chaque vraie demande à l’agent — le modèle que vous avez sélectionné répond avec l’ensemble des outils. Nécessite une clé API OpenAI ; la couche vocale est facturée 0,05 $ par minute.',
      'voice.gptLive.instructions':
        'Phrases supplémentaires pour la personnalité vocale en direct (ton, rythme, langue). L’agent conserve ses propres instructions système.',
      'display.personality': 'Style par défaut de l’assistant pour les nouvelles discussions.',
      'memory.memoryEnabled': 'Sauvegarder des mémoires durables pouvant aider les discussions futures.'
    },
    sessions: {
      loading: 'Chargement des discussions archivées…',
      archivedTitle: 'Discussions archivées',
      defaultDirDesc:
        'Les nouvelles discussions commencent dans ce dossier sauf si vous en choisissez un autre. Laissez-le non défini pour utiliser votre répertoire personnel.',
      failedLoad: 'Impossible de charger les discussions archivées'
    }
  },
  skills: {
    plugins: {
      agentBlurb:
        'Étendez l’agent sélectionné avec des outils, hooks et fournisseurs. Les changements prennent effet après le redémarrage de Hermes.',
      halfDesktopHint: 'cette application, identique pour tous les agents',
      halfAgentIn: profile => `Agent ${profile}`,
      defaultProfile: 'Agent par défaut',
      installAgentHereTip: profile =>
        `La partie Desktop est chargée dans cette application, mais la partie agent n’est pas installée pour ${profile}. Installez-la pour cet agent.`,
      installAgentHereNoOrigin:
        'La partie agent n’est pas installée pour cet agent et ce paquet a été copié manuellement, sans entrée de catalogue ni dépôt Git. Copiez son dossier dans celui de l’agent ou réinstallez-le depuis Git.',
      empty: 'Aucun plugin installé pour cet agent.',
      alreadyInstalled: (name: string) => `${name} est déjà installé pour cet agent.`,
      updated: (name: string) =>
        `${name} mis à jour vers l’épinglage actuel du catalogue. Redémarrez Hermes pour l’appliquer.`,
      autoUpdateTip: (hours: number) =>
        `Appliquer automatiquement les mises à jour des plugins lors de la vérification par Hermes (toutes les ${hours} h). Le nouveau code se charge dans les nouvelles discussions.`,
      uninstallConfirmBody: (name: string, profile: string) =>
        `Cette action supprime les fichiers du plugin de l’agent ${profile}. Toute moitié bureau fournie est supprimée avec lui. Réinstallez-le depuis le catalogue ou depuis Git à tout moment.`,
      uninstalled: (name: string) => `${name} désinstallé. Redémarrez Hermes pour le décharger.`,
      settingsForm: {
        secretStoredAs: (env: string) =>
          `Stocké dans le .env de l’agent sous ${env}, jamais dans config.yaml ; laissez vide pour conserver la valeur actuelle.`
      }
    },
    remotePluginHint:
      'Ouvrez cet agent sur sa machine pour gérer ses plugins. L’installation des plugins utilise la machine connectée.',
    openAgentSettings: 'Paramètres de l’agent',
    changesApplyNewSessions: 'Les modifications s’appliquent aux nouvelles discussions.',
    appliesToNewSessions: name => `${name} s’applique aux nouvelles discussions.`,
    bulkUpdated: count => `${count} ${count === 1 ? 'élément' : 'éléments'} mis à jour pour les nouvelles discussions.`,
    openToolLibrary: 'Parcourir le catalogue d’outils'
  },
  starmap: {
    emptyDesc:
      'Au fur et à mesure que l’agent construit des skills et des mémoires pour votre travail, ils apparaissent ici.'
  },
  gatewayGroupOpen: {
    lookupFailed: 'Impossible de retrouver ce groupe sur cette machine. Réessayez.'
  },
  commandCenter: {
    sectionEntries: {
      system: {
        detail: 'État de la machine, journaux, redémarrage/mise à jour'
      },
      sessions: {
        title: 'Panneau des discussions',
        detail: 'Rechercher, épingler et gérer les discussions'
      }
    },
    gatewayRunning: 'La messagerie est en cours d’exécution',
    gatewayStopped: 'La messagerie est arrêtée',
    restartGateway: 'Redémarrer Hermes',
    gatewayRestartFailed: 'Échec du redémarrage de Hermes.',
    sharedGatewayRestartTitle: 'Redémarrer Hermes pour tous les agents ?',
    sharedGatewayRestartDescription: bots => `Tous les agents de cet appareil se reconnecteront : ${bots}`,
    sharedGatewayRestarted: count => `Hermes redémarré (${count} ${count === 1 ? 'agent' : 'agents'})`,
    searchPlaceholder: 'Rechercher des discussions, vues et actions',
    goToSession: 'Aller à la discussion',
    nav: {
      newChat: {
        title: 'Nouvelle discussion',
        detail: 'Commencer une nouvelle discussion'
      },
      artifacts: {
        title: 'Bibliothèque'
      }
    },
    pinSession: 'Épingler la discussion',
    unpinSession: 'Désépingler la discussion',
    exportSession: 'Exporter la discussion',
    deleteSession: 'Supprimer la discussion',
    noSessions: 'Aucune discussion pour le moment.',
    maintenance: {
      backupDesc: 'Archiver en zip la configuration, les mémoires, les skills et les discussions'
    },
    newSessionInProject: project => `Nouvelle discussion dans ${project}`,
    hermesActiveSessions: (version, count) => `Hermes ${version} · Discussions actives ${count}`
  },
  messaging: {
    states: {
      gateway_stopped: 'La messagerie est arrêtée'
    },
    hintPendingRestart: "Redémarrez Hermes depuis la barre d'état pour appliquer ce changement.",
    sharedListenerUrl: 'Servi sur le point d’écoute partagé de Hermes à',
    hintGatewayStopped: "Démarrez Hermes depuis la barre d'état pour vous connecter.",
    gatewayStopped: 'La messagerie est arrêtée',
    restartToApply: 'Ce changement prend effet après un redémarrage de Hermes.',
    restartToReconnect: 'Les nouveaux identifiants prennent effet après un redémarrage de Hermes.',
    appliedLive: 'Appliqué à Hermes en cours d’exécution.',
    connectingLive: 'Hermes se connecte avec les nouveaux identifiants.',
    restartNeeded: 'Enregistré. Redémarrez Hermes pour appliquer les nouveaux paramètres.',
    restartFailedManual: 'Le redémarrage de Hermes a échoué — redémarrez-le manuellement et consultez ses journaux.',
    relayLink: {
      managedNote: (platform: string) =>
        `Team Gateway assure la messagerie sur cette instance : Hermes désactive donc la connexion directe ${platform} à chaque démarrage. Pour l’utiliser en plus du relais, définissez GATEWAY_RELAY_ALLOW_DIRECT_PLATFORMS=true dans l’environnement de Hermes (une connexion directe empêche l’instance de descendre à zéro et peut livrer les messages deux fois).`
    },
    telegramQr: {
      savedRestarting: 'Telegram enregistré ; redémarrage de Hermes…',
      savedRestartFailed: detail => `Telegram enregistré ; le redémarrage de Hermes a échoué${detail}`
    },
    fieldCopy: {
      DISCORD_HOME_CHANNEL: {
        help: 'Canal où le bot envoie des messages proactifs (résultats des routines, rappels).'
      },
      QQBOT_HOME_CHANNEL: {
        help: 'Canal ou groupe par défaut pour les résultats des routines.'
      }
    }
  },
  webhooks: {
    disabledBody:
      'Les webhooks fonctionnent indépendamment des canaux de conversation. Activez-les ici pour accepter les événements HTTP entrants ; les canaux de conversation ne sont nécessaires que lorsqu’un abonnement livre vers Telegram, Discord, Slack ou un autre canal.',
    restarting: 'Redémarrage de Hermes...',
    restartNeeded:
      "Les webhooks sont activés, mais Hermes a toujours besoin d'un redémarrage avant que le récepteur puisse se connecter.",
    restartGateway: 'Redémarrer Hermes',
    restartFailed: (detail: string) => `Échec du redémarrage de Hermes${detail}`,
    enabledRestarting: 'Webhooks activés ; redémarrage de Hermes...'
  },
  profiles: {
    close: 'Fermer les agents',
    title: 'Agents',
    count: count => `${count} ${count === 1 ? 'agent' : 'agents'}`,
    search: 'Rechercher des agents...',
    loading: 'Chargement des agents...',
    newProfile: 'Nouvel agent',
    importProfile: 'Importer un agent…',
    exportProfile: 'Exporter l’agent…',
    imported: 'Agent importé',
    exported: 'Agent exporté',
    failedImport: 'Échec de l’importation de l’agent',
    failedExport: 'Échec de l’exportation de l’agent',
    exportNeedsHost: name =>
      `Le navigateur ne peut pas enregistrer une archive d’agent. Sur la machine qui exécute Hermes, exécutez : hermes profile export ${name}`,
    allProfiles: 'Tous les agents',
    showAllProfiles: 'Afficher tous les agents',
    manageProfiles: 'Gérer les agents…',
    connectGateway: 'Gérer les machines…',
    fleet: {
      allOnGateway: 'Tous les agents de cette machine',
      gateway: gateway => `Agents sur ${gateway}`,
      localDevice: 'Cet appareil (backend local — installe Hermes s’il manque, sinon ouvre une nouvelle discussion)',
      installDeviceDesc:
        'Hermes sera installé localement, puis une nouvelle discussion s’ouvrira sur cet ordinateur. Rien n’est installé tant que vous ne confirmez pas.'
    },
    remoteOverride: {
      description:
        'Les discussions de cet agent s’exécuteront sur le Hermes distant que vous choisissez, plutôt que sur cet ordinateur.',
      collisionWarning: (label: string) =>
        `Une machine nommée « ${label} » existe déjà dans les paramètres. Cette connexion d’agent est indépendante et ne la modifiera pas.`,
      confirmTitle: 'Connecter cet agent à une machine distante ?',
      savedTitle: 'Agent connecté',
      menuItem: 'Se connecter à une machine distante…',
      authFailedTitle: 'La machine distante a refusé le jeton enregistré',
      title: (profile: string) => `Connecter ${profile} à une machine distante`,
      confirmNote: (profile: string, host: string) =>
        `Les nouvelles discussions de ${profile} s’exécuteront sur ${host}. Cette machine exécutera les commandes et lira les fichiers qui s’y trouvent, pas ceux de cet appareil. Connectez-vous uniquement à une machine de confiance.`
    },
    defaultProfile: 'Agent par défaut',
    defaultSet: name => `${name} est maintenant l’agent par défaut`,
    defaultDescription:
      'Utilisé à l’ouverture de Hermes et pour les nouvelles discussions. Les discussions existantes restent avec leur agent.',
    failedSetDefault: 'Impossible de définir l’agent par défaut',
    noProfiles: 'Aucun agent pour le moment.',
    selectPrompt: 'Sélectionnez un agent pour afficher ses détails.',
    refresh: 'Actualiser les agents',
    refreshing: 'Actualisation des agents',
    soulDesc: 'Les instructions et la personnalité de cet agent.',
    soulPlaceholder: mode =>
      `Les instructions et la personnalité de cet agent.\nLaissez vide pour conserver la valeur par défaut ${mode}.`,
    deleteTitle: 'Supprimer l’agent ?',
    createDesc: 'Chaque agent possède ses propres instructions, paramètres, skills et discussions.',
    cloneFromDesc: 'Copie la configuration, les skills et le SOUL.md depuis l’agent source sélectionné.',
    cloneFromDefault: 'Cloner depuis l’agent par défaut',
    cloneFromDefaultDesc: 'Copier la configuration, les skills et le SOUL.md depuis votre agent par défaut.',
    createAction: 'Créer l’agent',
    renameTitle: 'Renommer l’agent',
    renameDescPrefix: 'Le renommage met à jour le répertoire de l’agent et les scripts d’enveloppe dans ',
    displayNameDesc:
      'Définit le nom affiché dans toute l’application. L’identifiant interne de l’agent reste « default ».',
    created: 'Agent créé',
    renamed: 'Agent renommé',
    deleted: 'Agent supprimé',
    failedLoad: 'Échec du chargement des agents',
    failedDelete: 'Échec de la suppression de l’agent',
    failedCreate: 'Échec de la création de l’agent',
    failedRename: 'Échec du renommage de l’agent',
    status: {
      unread: (count: number) => (count === 1 ? '1 discussion non lue' : `${count} discussions non lues`),
      needsInput: (count: number) =>
        count === 1 ? '1 discussion attend votre réponse' : `${count} discussions attendent votre réponse`,
      working: (count: number) => (count === 1 ? '1 discussion en cours' : `${count} discussions en cours`)
    }
  },
  cron: {
    close: 'Fermer les routines',
    title: 'Routines',
    count: count => `${count} ${count === 1 ? 'routine' : 'routines'}`,
    search: 'Rechercher des routines...',
    loading: 'Chargement des routines...',
    editJob: 'Modifier la routine',
    newCron: 'Nouvelle routine',
    emptyDescNew:
      'Les routines exécutent automatiquement vos instructions et envoient les résultats à la destination de votre choix.',
    emptyTitleNew: 'Aucune routine pour le moment',
    actionsTitle: 'Actions de la routine',
    resume: 'Reprendre la routine',
    pause: 'Mettre en pause la routine',
    edit: 'Modifier la routine',
    deleteTitle: 'Supprimer la routine ?',
    resumed: 'Routine reprise',
    paused: 'Routine en pause',
    triggered: 'Routine déclenchée',
    deleted: 'Routine supprimée',
    created: 'Routine créée',
    updated: 'Routine mise à jour',
    failedLoad: 'Échec du chargement des routines',
    failedUpdate: 'Échec de la mise à jour de la routine',
    failedTrigger: 'Échec du déclenchement de la routine',
    failedDelete: 'Échec de la suppression de la routine',
    failedSave: 'Échec de l’enregistrement de la routine',
    editTitle: 'Modifier la routine',
    createTitle: 'Nouvelle routine',
    editDesc:
      'Modifiez la fréquence, les instructions ou la destination des résultats. Les modifications s’appliquent à la prochaine exécution.',
    createDesc:
      'Exécutez automatiquement des instructions à la fréquence choisie. Utilisez la syntaxe cron ou une phrase comme « toutes les 15 minutes ».',
    customScheduleLabel: 'Fréquence personnalisée',
    promptScheduleRequired: 'L’invite et la fréquence sont requises.',
    scheduleRequired: 'La fréquence est requise.',
    scriptOnlyEditHint: 'Routine exécutant un script uniquement (pas d’invite IA). ID de la routine :',
    createAction: 'Créer la routine',
    tabs: {
      jobs: 'Routines'
    },
    blueprints: {
      dialogDesc: 'Remplissez les détails et choisissez quand elle s’exécute.',
      scheduled: 'Routine créée à partir du modèle'
    }
  },
  sidebar: {
    filter: {
      profile: 'Agent'
    },
    gatewayGroups: {
      grouping: 'Machine et agent',
      aliasHint: 'Seul le nom d’affichage change ; les noms de la machine et de l’agent restent inchangés.'
    },
    profileRail: 'Barre des agents',
    nav: {
      cron: 'Routines',
      'new-session': 'Nouvelle discussion',
      artifacts: 'Bibliothèque'
    },
    cronJobs: 'Routines',
    storageCorrupt: {
      action: 'Arrêtez cet agent, puis inspectez le fichier sans le modifier, ou restaurez un instantané :',
      title: 'La base de données des discussions est endommagée',
      body: (profiles: string) =>
        `Hermes ne peut pas lire tout l’historique des discussions de ${profiles}. Les discussions absentes de cette liste n’ont pas été supprimées ; le fichier qui les contient est endommagé.`
    },
    projects: {
      createdInPreviousContext:
        'Le projet a été créé sur la connexion ou l’agent précédent. Revenez-y ; IDEA.md n’a pas été écrit.',
      showAllSessions: 'Afficher toutes les discussions',
      moveFailed: 'Impossible de déplacer la discussion',
      toggle: (label, open) => `${open ? 'Afficher' : 'Masquer'} les discussions de ${label}`,
      showAllCount: count => `Afficher les ${count} discussions`
    },
    row: {
      ownedByProfile: profile => `Agent : ${profile}`,
      copyIdFailed: 'Impossible de copier l’ID de discussion',
      sessionActions: 'Actions de discussion',
      sessionRunning: 'Discussion en cours',
      renameTitle: 'Renommer la discussion',
      untitledPlaceholder: 'Discussion sans titre',
      deleteTitle: 'Supprimer la discussion ?',
      deleted: 'Discussion supprimée'
    },
    searchAria: 'Rechercher des discussions',
    searchPlaceholder: 'Rechercher des discussions…',
    groupAriaGrouped: 'Afficher les discussions en une seule liste',
    groupAriaUngrouped: 'Grouper les discussions par espace de travail',
    showSessions: 'Afficher les discussions',
    groupTitleGrouped: 'Dégrouper les discussions',
    projectEmpty: 'Aucune discussion pour le moment',
    projectLoadFailed: 'Impossible de charger les discussions',
    noSessions: 'Aucune discussion pour le moment',
    noFilterMatches: 'Aucune discussion ne correspond à ces filtres',
    noMatch: query => `Aucune discussion ne correspond à « ${query} ».`,
    newSessionIn: label => `Nouvelle discussion dans ${label}`
  },
  composer: {
    placeholderStarting: 'Démarrage de l’agent…',
    placeholderReconnecting: 'Reconnexion à l’agent…',
    voiceEngineChained: 'Reconnaissance vocale + voix de l’agent',
    voiceEngineLive: 'GPT-Live (duplex intégral, délègue à l’agent)',
    commandDescs: {
      '/export': 'Exporter un agent (configuration, skills, thème) dans une archive partageable',
      '/import': 'Importer une archive d’agent partagée comme nouvel agent',
      '/goal':
        'Définir un objectif permanent sur lequel l’agent travaille au fil des tours jusqu’à ce qu’il soit atteint',
      '/profile': 'Changer d’agent actif',
      '/platform': 'Suspendre, reprendre ou lister une plateforme de messagerie en échec',
      '/clear': 'démarrer une nouvelle discussion',
      '/resume': 'Reprendre une discussion enregistrée',
      '/topic': 'Activer ou inspecter les discussions par sujet des MP Telegram',
      '/title': 'Renommer la discussion actuelle',
      '/handoff': 'Transférer cette discussion vers une plateforme de messagerie',
      '/bg': 'Exécuter un prompt dans une discussion d’arrière-plan distincte',
      '/heartbeat': 'Définir un prompt récurrent qui revient dans cette discussion lorsqu’elle est inactive',
      '/loop': 'Relancer un prompt à intervalle régulier dans cette discussion',
      '/status': 'Afficher l’état de la discussion actuelle',
      '/model': 'Changer le modèle de cette discussion'
    },
    attachUrlDesc: 'L’agent récupérera la page et l’inclura comme contexte pour ce tour.',
    queueLostDiscardTip:
      "Hermes a redémarré en plein tour et ne peut pas le terminer. Ignorez-le pour que les tours en file d'attente derrière lui s'exécutent.",
    newSessionPlaceholders: [
      'Sur quoi travaillons-nous ?',
      'Donnez une tâche à l’agent',
      "Qu'avez-vous en tête ?",
      'Décrivez ce dont vous avez besoin',
      "Qu'est-ce qu'on attaque ?",
      "Posez n'importe quelle question",
      'Commencez par un objectif'
    ],
    queueDroppedBody:
      'Cette entrée d’arrière-plan a été abandonnée car sa discussion n’a pas pu être reprise après plusieurs tentatives. Le reste de la file d’attente est inchangé.',
    browseAllSessions: 'Parcourir toutes les discussions…'
  },
  statusStack: {
    coding: {
      agentShip: 'Demander à l’agent d’ouvrir une PR'
    },
    control: {
      loopDeferredNotice: 'Un objectif actif contrôle actuellement la discussion.',
      continuationBusy:
        'Objectif repris — discussion occupée, arrêtez d’abord la réponse en cours (bouton Stop ou Échap) pour poursuivre',
      waitSession: target => `En attente de la discussion ${target}`,
      controlUnavailable: msg => `Commandes de discussion indisponibles : ${msg}`
    }
  },
  updates: {
    bundleOutOfSyncDesc:
      "Le runtime Hermes a été mis à jour, mais l'application Desktop utilise encore une ancienne version. Les nouvelles fonctions de l'interface, comme le mode Agent, resteront absentes jusqu'à sa mise à jour. Lancez la mise à jour ci-dessous pour reconstruire l'application. Si cet avertissement persiste, réinstallez-la avec le dernier installateur Desktop.",
    applyStatus: {
      noReturn:
        'Le backend ne s’est pas reconnecté. La mise à jour n’est peut-être pas terminée — vérifiez la machine qui exécute Hermes.'
    }
  },
  install: {
    setupChoiceDesc:
      'Connectez cette application à une machine sur laquelle Hermes fonctionne déjà, ou installez Hermes localement sur cet ordinateur.',
    remoteSetupDesc:
      'Entrez l’URL de votre machine. Hermes Desktop détectera s’il a besoin d’un jeton ou d’une connexion par navigateur.',
    remoteUrlTitle: 'URL de la machine',
    remoteUrlDesc: 'Utilisez l’URL de base de Hermes sur la machine, y compris https:// pour les connexions distantes.',
    probing: 'Détection de l’authentification de la machine...',
    probeError: "Impossible d'atteindre cet ordinateur avec Hermes.",
    authNeedsOauth: provider => `Connectez-vous avec ${provider} avant de tester la connexion à cette machine.`,
    enterUrlFirst: 'Entrez d’abord une URL de machine.',
    tokenDesc: 'Collez le jeton de session issu du fichier .env de Hermes sur la machine distante.',
    incompleteSignInTest: 'Connectez-vous avant de tester la connexion protégée par OAuth à cette machine.',
    incompleteTokenTest: 'Entrez un jeton de session avant de tester la connexion à cette machine.',
    connectExistingDesc:
      'Utilisez une machine distante avec un jeton de session ou une connexion par navigateur. Aucune installation locale ne démarrera.'
  },
  onboarding: {
    headerTitle: 'Configurons votre agent'
  },
  shell: {
    gatewayMenu: {
      gateway: 'Ordinateur',
      reconnectGateway: 'Reconnecter la machine',
      machineSettings: 'Machines',
      restartTitle: 'Redémarrer Hermes sur cette machine ?'
    },
    statusbar: {
      gateway: 'Ordinateur',
      gatewayRestarting: 'redémarrage de Hermes…',
      gatewayTitle: 'Ordinateur',
      openCron: 'Ouvrir les routines',
      toggleSessionTimer: 'Minuteur de discussion',
      cacheHitRateTitle:
        'Taux de cache des prompts pour cette discussion — les tokens en cache coûtent moins cher, un taux élevé est donc plus économique'
    }
  },
  rightSidebar: {
    terminalGatewayRemoved: 'Machine supprimée',
    terminalGatewayRemovedBody:
      'La machine sur laquelle ce terminal tournait a été supprimée ; le terminal ne peut pas être rouvert. Fermez cet onglet.',
    terminalGatewayNoShellBody:
      'Cette machine n’offre pas d’accès shell ; ce terminal ne peut pas être rouvert. Fermez cet onglet.'
  },
  preview: {
    saveScopeChanged: 'Revenez à la connexion et à l’agent d’origine pour enregistrer ce brouillon.',
    web: {
      restarting: 'L’agent redémarre le serveur...',
      askRestart: 'Demander à l’agent de redémarrer le serveur',
      lookingRestart: taskId => `L’agent recherche un serveur d’aperçu à redémarrer (${taskId})`,
      restartingMessage:
        'L’agent travaille en arrière-plan. Surveillez la console d’aperçu pour suivre la progression.',
      blankPageBody: 'Saisissez une adresse ci-dessus pour naviguer, ou demandez à l’agent d’ouvrir une page.',
      finishedRestarting: message =>
        `L’agent a terminé le redémarrage du serveur d’aperçu${message ? `: ${message}` : ''}`,
      restartFailedMessage: 'L’agent n’a pas pu redémarrer le serveur.',
      stillWorking:
        'L’agent travaille toujours, mais aucun résultat de redémarrage n’est arrivé. La commande du serveur peut être en cours d’exécution au premier plan.'
    }
  },
  interfaceMode: {
    hint: 'Modifie ce qui est affiché, pas ce que l’agent peut faire.',
    simple: {
      description:
        'Pour discuter avec l’agent. Barre latérale et discussion ; pas de panneaux terminal, fichiers ou diff.'
    }
  },
  assistant: {
    thread: {
      loadingResponse: 'L’agent charge une réponse',
      errorLayerBodies: {
        generic:
          'Une erreur s’est produite pendant la réponse de l’agent. Réessayez ou copiez les détails si cela persiste.'
      },
      errorCodes: {
        timeout: {
          body: provider => `L’agent n’a pas pu joindre ${provider}. Vérifiez votre connexion, puis réessayez.`
        },
        no_reply: {
          body: 'L’agent a terminé ce tour, mais aucune réponse n’est parvenue à cette fenêtre. Réessayez pour la renvoyer.'
        },
        cron_session_read_only: {
          title: 'Cette discussion appartient à une routine',
          body: 'Elle sert uniquement à cette routine ; votre message n’a donc pas été exécuté. Démarrez une nouvelle discussion pour l’envoyer.'
        },
        agent_init_failed: {
          body: provider =>
            `L’agent n’a pas pu se connecter à ${provider} pour cette conversation. Réessayez ; si l’échec persiste, vérifiez le fournisseur dans Paramètres.`
        },
        free_tier_model_not_free: {
          body: 'L’agent utilise le modèle gratuit pour le moment. Connectez-vous avec un compte Nous gratuit pour accéder à plus de modèles.'
        },
        launch_credentials_unavailable: {
          body: 'Cette discussion a démarré avec une clé API qui a depuis été modifiée ou supprimée ; votre message n’a donc pas été envoyé. Une discussion ouverte ne passe jamais d’elle-même à une autre clé. Utilisez « Démarrer une nouvelle discussion » pour utiliser votre clé actuelle.'
        }
      },
      errorToastTitle: 'L’agent n’a pas pu terminer la réponse',
      loadingSession: 'Chargement de la discussion',
      errorStartNewSession: 'Démarrer une nouvelle discussion'
    },
    approval: {
      gatewayDisconnected: "Hermes n'est pas connecté",
      timedOutSystemLine:
        'Le délai d’approbation a expiré — la commande n’a pas été exécutée. Demandez à l’agent de réessayer ou augmentez la limite dans Paramètres → Sécurité → Délai d’approbation.',
      alwaysDescription: pattern =>
        `Cela ajoute le motif « ${pattern} » à votre liste d’autorisation permanente (~/.hermes/config.yaml). L’agent ne redemandera plus pour ce type de commande — que ce soit dans cette discussion ou dans une discussion future.`,
      allowSession: 'Autoriser pour cette discussion'
    },
    clarify: {
      gatewayDisconnected: "Hermes n'est pas connecté"
    },
    themePicker: {
      basicDetail: 'Pour discuter avec l’agent.'
    },
    catalogInstall: {
      targetProfile: (profile: string) => `S’installe pour votre agent ${profile}`
    },
    mcpSetup: {
      gatewayDisconnected: "Hermes n'est pas connecté",
      reloadFailed:
        'Serveur enregistré, mais le rechargement des outils MCP a échoué — ils seront chargés dans la prochaine discussion'
    },
    tool: {
      titles: {
        cronjob: {
          done: 'Routine',
          pending: 'Création de la routine en cours',
          pendingAction: 'Configuration en cours'
        },
        session_search_recall: {
          done: 'Historique des discussions recherché',
          pending: 'Recherche dans l’historique des discussions en cours'
        }
      }
    }
  },
  prompts: {
    gatewayDisconnected: "Hermes n'est pas connecté",
    sudoDesc:
      'L’agent a besoin de votre mot de passe sudo pour exécuter une commande privilégiée. Il n’est envoyé qu’à votre agent local.',
    sudoInstallDesc:
      'Hermes a besoin de votre mot de passe sudo pour installer les paquets de l’écran de l’agent (TigerVNC + Xfce) sur sa machine. Il n’est envoyé qu’à cette machine.',
    secretDesc: 'L’agent a besoin d’un identifiant pour continuer.',
    vaultSaveDesc: origin =>
      `L’agent a atteint une page de connexion sur ${origin} et ne possède aucun identifiant pour celle-ci. Saisissez-le une fois ici : il sera chiffré sur cette machine et rempli dans la page sans que le modèle voie le mot de passe.`,
    vaultCodeDesc: site =>
      `${site} demande un code à usage unique reçu par SMS, e-mail ou application d’authentification. Saisissez-le ici : l’agent le remplira dans la page sans que le modèle le voie.`,
    vaultCodeFootnote:
      'Astuce : enregistrez la clé d’authentification avec cet identifiant dans Paramètres → Mots de passe et identifiants ; l’agent saisira alors les codes pour vous.',
    vaultUnlockDesc: name =>
      `L'agent souhaite se connecter à un site avec un identifiant enregistré dans ${name}. Saisissez votre mot de passe principal pour le déverrouiller pendant cette session : il est envoyé directement à ${name} sur cette machine, sans être enregistré ni montré à l'agent.`
  },
  desktop: {
    profileStatus: current =>
      `Agent : ${current}. Utilisez /profile <name> ou le sélecteur « Nouvelle discussion » pour démarrer une discussion avec un autre agent.`,
    unknownProfile: 'Agent inconnu',
    noProfileNamed: (target, available) => `Aucun agent nommé « ${target} ». Disponibles : ${available}`,
    newChatsProfile: name => `Les nouvelles discussions utiliseront l’agent ${name}.`,
    setProfileFailed: 'Échec de la sélection de l’agent',
    resumeStrandedBody:
      'La connexion à cette discussion a échoué et les nouvelles tentatives automatiques ont été abandonnées. Vérifiez que Hermes fonctionne sur la machine, puis réessayez.',
    poolSlotTimeoutBody:
      'Tous les emplacements de backend local sont occupés. Augmentez le nombre de backends d’agents maintenus actifs dans Paramètres → Avancé, ou réessayez après l’éviction d’un backend inactif.',
    handoff: {
      timedOut:
        'Hermes n’a pas pu joindre votre connexion de messagerie. Démarrez-la depuis Paramètres → Messagerie, puis réessayez le transfert.'
    },
    sessionUnavailable: 'Discussion indisponible',
    createSessionFailed: 'Impossible de créer une nouvelle discussion',
    resumeStrandedTitle: 'Impossible de charger cette discussion',
    sessionBusy: 'Discussion occupée',
    cwdStagedMessage:
      'Redémarrez le backend desktop pour appliquer les modifications de cwd à cette discussion active.',
    sessionExported: 'Discussion exportée',
    sessionExportFailed: 'Impossible d’exporter la discussion',
    yoloSystem: active => `YOLO ${active ? 'on' : 'off'} pour cette discussion`,
    sessionOpenFailed: 'Impossible d’ouvrir cette discussion'
  },
  tips: {
    items: {
      skills: {
        text: 'Les compétences sont des dossiers d’instructions que l’agent charge lorsque le travail le nécessite.'
      },
      messaging: {
        title: 'Votre agent loin de votre bureau'
      },
      artifacts: {
        title: 'Tout ce que l’agent a créé',
        text: 'Images, fichiers et liens de chaque discussion, réunis dans votre bibliothèque.'
      },
      cron: {
        text: 'Les routines exécutent une invite toutes les heures, chaque nuit ou selon une expression cron.'
      },
      profiles: {
        title: 'Chaque agent est indépendant',
        text: 'Chacun possède ses propres instructions, paramètres, mémoire et discussions.'
      },
      'command-palette': {
        text: 'Discussions, paramètres, compétences et commandes sont accessibles depuis la palette.'
      }
    }
  },
  intro: {
    stock: {
      none: [
        'Posez une question, collez une erreur ou indiquez-moi un dépôt. Je peux lire du code, utiliser des outils et vous aider à livrer.',
        'Décrivez la tâche avec vos mots. Je choisis les bons outils, explique mon plan et vous consulte avant les étapes risquées.',
        'Indiquez un chemin de fichier, une traceback ou une idée brute. J’enquête, propose les étapes suivantes et garde tout réversible.',
        'Cherchez dans le dépôt, modifiez des fichiers, lancez les tests, ouvrez des PR. Donnez-moi l’objectif, je gère la partie mécanique.',
        'Saisissez une tâche, une question ou un extrait. Je garde en mémoire la discussion, cite mes sources et m’arrête pour demander en cas de doute.'
      ]
    }
  },
  sessionImport: {
    action: 'Importer une discussion',
    search: 'Rechercher parmi les discussions chargées',
    scanError: 'Impossible de trouver les discussions',
    emptyHelp: 'Les discussions Claude Code et Codex de cette machine apparaîtront ici.',
    searchHelp: 'Essayez un autre titre ou dossier, ou chargez davantage de discussions.',
    more: 'Charger davantage de discussions',
    chooseHelp: 'Choisissez une discussion pour lire son historique avant de l’importer dans Hermes.'
  },
  titlebar: {
    searchTitle: 'Rechercher des discussions, vues et actions',
    unreadSessions: count => (count === 1 ? '1 discussion non lue' : `${count} discussions non lues`)
  },
  artifacts: {
    search: 'Rechercher dans la bibliothèque...',
    refresh: 'Actualiser la bibliothèque',
    refreshing: 'Actualisation de la bibliothèque',
    indexing: 'Indexation des sorties de discussion récentes',
    noArtifactsTitle: 'Rien dans votre bibliothèque pour le moment',
    noArtifactsDesc:
      'Les images générées et les sorties de fichier apparaîtront ici au fur et à mesure que les discussions les produisent.',
    failedLoad: 'Échec du chargement de la bibliothèque',
    partialIndex: (failed, total) =>
      `${failed} discussions récentes sur ${total} ignorées lors de l’indexation de la bibliothèque.`
  },
  artifactPreview: {
    missingTitle: 'Élément indisponible',
    missingBody: 'Cet élément ne figure plus dans la bibliothèque.'
  },
  freeTier: {
    unreachableBody:
      'Hermes n’a pas pu joindre le service Nous pour terminer votre connexion. Vérifiez votre connexion Internet et réessayez. Votre discussion reste disponible.',
    setupFailed: {
      locked:
        'Cette discussion ne peut pas continuer sans connexion. Connectez-vous ou créez un compte Nous gratuit pour poursuivre.'
    },
    busyBody: wait =>
      `Hermes n’a pas pu terminer votre connexion car le service Nous est occupé. Réessayez dans ${wait}. Votre discussion reste disponible entre-temps.`
  },
  zones: {
    closeRunningBody:
      'Cette discussion est toujours en cours (ou en attente de votre saisie). Fermer l’onglet la masque — la discussion conserve sa progression et peut être rouverte depuis la barre latérale.',
    newSessionTab: 'Nouvel onglet de discussion'
  }
}
