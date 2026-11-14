import type { GatewayHostTranslations } from './types_gateway_host'

// Settings → Machines copy where the connection cannot change, spread into fr.ts.
export const frGatewayHost: GatewayHostTranslations = {
  unavailableTitle: 'Paramètres du gateway indisponibles',
  unavailableDesc: "Le pont IPC du desktop n'expose pas les paramètres du gateway.",
  webappHostTitle: 'Hôte Hermes',
  webappHostDesc:
    'La Webapp s’exécute toujours sur l’hôte Hermes qui la sert. Les connexions enregistrées ci-dessous sont la liste de cette machine, partagée avec l’app Hermes Desktop ; les autres passerelles sont atteintes via cet hôte.',
  webappOauthPaste: {
    title: 'Terminer la connexion',
    description:
      'Si l’onglet de connexion s’est arrêté sur une adresse qui ne charge pas (ce navigateur n’est pas sur la machine de la passerelle), collez cette adresse ici.',
    placeholder: 'http://127.0.0.1:…/webapp/gateway-connections/oauth/callback?…',
    finish: 'Terminer',
    failed: 'Impossible de terminer la connexion'
  },
  webAccess: {
    title: 'Accès web',
    intro:
      'Ouvrez Hermes sur cet ordinateur depuis votre téléphone ou un autre ordinateur, avec votre compte Nous. Toute personne connectée peut exécuter des commandes sur cet ordinateur.',
    failed: "Impossible de modifier l'accès web",
    running: 'En cours',
    stopped: 'Arrêté',
    starting: 'Démarrage',
    stopping: 'Arrêt en cours',
    restartNeeded: 'Redémarrage requis',
    unavailable: 'Indisponible',
    notRunningDetail: 'Démarrez-le pour ouvrir Hermes depuis un autre appareil.',
    needsNous: "Connectez-vous d'abord à votre compte Nous dans Fournisseurs.",
    runningLan: 'Ouvrez cette adresse sur un appareil du même réseau.',
    runningPublic: 'Servi via votre URL.',
    runningLocal:
      "Seul cet ordinateur peut ouvrir l'application web. Arrêtez-la, puis relancez-la pour l'ouvrir depuis d'autres appareils.",
    pendingOn: "Enregistré. La passerelle servira l'application web après son redémarrage.",
    pendingOff: "Désactivé, mais la passerelle continue de servir l'application web jusqu'à son redémarrage.",
    unavailableDetail: reason => `La passerelle n'a pas pu servir l'application web : ${reason}`,
    restartBusy:
      "Enregistré. Une conversation est en cours, la passerelle n'a donc pas redémarré. Choisissez « Redémarrer la passerelle » une fois qu'elle est terminée.",
    restartFailed: "Enregistré, mais la passerelle n'a pas pu redémarrer. Choisissez « Redémarrer la passerelle » pour réessayer.",
    restartFailedBecause: reason => `La passerelle n'a pas pu redémarrer : ${reason}`,
    startTimedOut: "La passerelle n'a pas servi l'application web en 90 secondes.",
    stopTimedOut: "La passerelle n'a pas cessé de servir l'application web en 90 secondes.",
    startStages: {
      preparing: "Préparation de l'application web. Sa compilation peut prendre quelques minutes.",
      building: "Compilation de l'application web. Cela peut prendre quelques minutes.",
      registering: 'Configuration de la connexion Nous pour cette adresse.',
      restarting: "Redémarrage de la passerelle pour servir l'application web."
    },
    startElapsed: time => `Temps écoulé : ${time}`,
    start: 'Démarrer',
    stop: 'Arrêter',
    restartGateway: 'Redémarrer la passerelle',
    modeLan: 'Ce réseau',
    modeLanDetail: url =>
      `Les appareils de votre Wi-Fi ouvrent ${url}. À utiliser uniquement sur un réseau de confiance.`,
    lanUnencrypted:
      "La connexion n'est pas chiffrée : d'autres personnes sur ce réseau pourraient lire ce que vous envoyez et recevez.",
    noLan: "Cet ordinateur n'a aucune adresse réseau accessible aux autres appareils.",
    modePublic: 'Ma propre URL',
    modePublicDetail: port =>
      `Une adresse HTTPS que vous gérez déjà, comme Tailscale Serve ou un tunnel, redirigée vers http://127.0.0.1:${port}.`
  }
}
