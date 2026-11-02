import type { PluginsTabCopy } from './types_plugins_tab'

export const frPluginsTab: PluginsTabCopy = {
  desktopHalfPending: 'copie…',
  desktopHalfPendingTip:
    "Ce paquet contient une partie Desktop qui n'a pas encore été copiée dans l'application. Relancez l'analyse ou redémarrez l'application.",
  desktopHalfRemote: 'indisponible (backend distant)',
  desktopHalfRemoteTip:
    'La moitié bureau de ce paquet se trouve sur le disque du backend distant, que cette application ne peut pas lire. Pour l’utiliser ici, lancez Installer depuis Git avec l’URL du dépôt du paquet et la cible Bureau cochée — cela clone la moitié bureau sur cette machine.',
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
  fleet: {
    updateEverywhere: (count: number) => `Mettre à jour partout · ${count}`,
    updateHereOnly: (profile: string) => `Seulement ${profile}`,
    updateAll: (count: number) => `Tout mettre à jour · ${count}`,
    updateAllTip: (installs: number, plugins: number) =>
      `${plugins === 1 ? '1 plugin' : `${plugins} plugins`} obsolètes dans ${installs === 1 ? '1 profil' : `${installs} profils`} sur vos gateways`,
    titleOne: (name: string) => `Mettre à jour ${name} partout`,
    titleAll: 'Mettre à jour les plugins partout',
    summary: (installs: number, gateways: number) =>
      `${installs === 1 ? '1 profil' : `${installs} profils`} sur ${gateways === 1 ? '1 gateway' : `${gateways} gateways`} utilisent une ancienne version.`,
    summaryDone: (installs: number) =>
      `${installs === 1 ? '1 profil mis à jour' : `${installs} profils mis à jour`}. Redémarrez ces gateways pour charger la nouvelle version.`,
    summaryFailed: (ok: number, failed: number) =>
      `${ok} mis à jour, ${failed} en échec. Survolez un échec pour en voir la raison.`,
    summaryDeclined: (ok: number, declined: number) =>
      `${ok} mis à jour. ${declined} non appliqués, les nouvelles autorisations ayant été refusées.`,
    thisProfile: '· ce profil',
    failed: 'Échec',
    declined: 'Non appliqué',
    skippedOffline: 'Hors ligne · ignoré',
    skippedSignIn: 'Déconnecté · ignoré',
    skippedIdle: 'Non connecté · ignoré',
    skippedUnreadable: 'Vérification impossible',
    confirm: (installs: number) => `Mettre à jour ${installs}`,
    updating: 'Mise à jour…',
    done: 'Terminé',
    consentBody: (names: string, installs: number) =>
      `La nouvelle version catalogue de ${names} ajoute des capacités absentes de la version installée. L’approbation s’applique à ${installs === 1 ? '1 profil' : `${installs} profils`} :`,
    finished: (installs: number) =>
      `${installs === 1 ? '1 profil mis à jour' : `${installs} profils mis à jour`}. Redémarrez ces gateways pour l’appliquer.`
  }
}
