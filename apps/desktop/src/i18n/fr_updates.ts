import type { TranslationOverrides } from './define-locale'

// The updates surface's copy (About page, updates overlay, statusbar version items), composed by fr.ts.
export const frUpdates = {
  updates: {
    discontinuedTitle: "Cette version de Hermes n'est plus prise en charge",
    discontinuedBody:
      "Cette version de Hermes n'est plus prise en charge et risque de ne plus fonctionner — désinstallez-la. Vos données restent sur le disque.",
    channels: { stable: 'Stable', canary: 'Canary' },
    appName: 'Hermes',
    availableBodyRelease: tag => `La version ${tag} est prête à être installée.`,
    releaseAvailable: tag => `La version ${tag} est disponible.`,
    checkingShort: 'Vérification…',
    availableBodyAppInstaller:
      'Une nouvelle version de Hermes est prête. Hermes va se fermer, Windows terminera la mise à jour, puis Hermes rouvrira automatiquement.',
    applyingBodyAppInstaller:
      "Hermes va se fermer et Windows terminera la mise à jour. Hermes rouvrira ensuite automatiquement — vous n'avez rien à faire.",
    applyingCloseAppInstaller:
      'Cette fenêtre va se fermer, Windows terminera la mise à jour et Hermes rouvrira automatiquement.',
    checkUnknownTitleAppInstaller: 'Impossible de vérifier les mises à jour',
    checkUnknownBodyAppInstaller:
      "Windows n'a pas pu rechercher les mises à jour. Elles s'installent également automatiquement au redémarrage de Hermes.",
    versionDetailsTitle: 'Détails de la version',
    versionDetailsBody:
      "Cette installation est gérée hors de l'application. Mettez-la à jour de la même manière que vous l'avez installée.",
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
    localBranchBehind: count =>
      `Cette copie de travail est sur une branche locale avec ses propres commits : ${count} commit${count === 1 ? '' : 's'} derrière le main upstream. Mettez-la à jour depuis un terminal avec \`hermes update\`.`,
    localBranchBehindUnknown:
      'Cette copie de travail est sur une branche locale avec ses propres commits ; sa distance par rapport au main upstream n’a pas pu être comptée.',
    localBranchCurrent: 'Cette copie de travail est sur une branche locale avec ses propres commits et est au niveau du main upstream.',
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
    copyFullLog: 'Copier le journal complet des modifications',
    manualTitle: 'Mise à jour depuis votre terminal',
    manualUnavailableTitle: 'Mise à jour impossible ici',
    manualBody:
      "Vous avez installé Hermes depuis la ligne de commande, les mises à jour s'y effectuent donc aussi. Collez ceci dans votre terminal :",
    manualPickedUp: 'Hermes prendra en compte la nouvelle version au prochain lancement.',
    manualBodyBackend:
      'Le backend Hermes est géré en dehors de cette app. Exécutez ceci sur le serveur qui l’héberge :',
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
  }
} satisfies Pick<TranslationOverrides, 'updates'>
