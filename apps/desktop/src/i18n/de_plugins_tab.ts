import type { PluginsTabCopy } from './types_plugins_tab'

export const dePluginsTab: PluginsTabCopy = {
  desktopHalfPending: 'wird kopiert…',
  desktopHalfPendingTip:
    'Dieses Paket enthält eine Desktop-Hälfte, die noch nicht in die App kopiert wurde. Verwenden Sie „Erneut scannen“ oder starten Sie die App neu.',
  desktopHalfRemote: 'nicht verfügbar (Remote-Backend)',
  desktopHalfRemoteTip:
    'Die Desktop-Hälfte dieses Pakets liegt auf der Festplatte des Remote-Backends, die diese App nicht lesen kann. Um sie hier zu nutzen, führen Sie „Aus Git installieren“ mit der Repository-URL des Pakets und aktiviertem Desktop-Ziel aus – dadurch wird die Desktop-Hälfte auf diesen Rechner geklont.',
  serverStates: {
    connected: 'verbunden',
    app_not_running: 'App läuft nicht',
    hermes_not_connected: 'MCP-Verbindung fehlt',
    endpoint_unavailable: 'Endpunkt nicht verfügbar',
    no_interactive_session: 'keine interaktive Session',
    version_too_old: 'Version zu alt',
    missing_app: 'App fehlt',
    unsupported_gpu: 'GPU nicht unterstützt',
    unknown: 'Status unbekannt'
  },
  fleet: {
    updateEverywhere: (count: number) => `Überall aktualisieren · ${count}`,
    updateHereOnly: (profile: string) => `Nur ${profile}`,
    updateAll: (count: number) => `Alle aktualisieren · ${count}`,
    updateAllTip: (installs: number, plugins: number) =>
      `${plugins === 1 ? '1 Plugin' : `${plugins} Plugins`} veraltet in ${installs === 1 ? '1 Profil' : `${installs} Profilen`} auf deinen Gateways`,
    titleOne: (name: string) => `${name} überall aktualisieren`,
    titleAll: 'Plugins überall aktualisieren',
    summary: (installs: number, gateways: number) =>
      `${installs === 1 ? '1 Profil' : `${installs} Profile`} auf ${gateways === 1 ? '1 Gateway' : `${gateways} Gateways`} nutzen eine ältere Version.`,
    summaryDone: (installs: number) =>
      `${installs === 1 ? '1 Profil' : `${installs} Profile`} aktualisiert. Starte diese Gateways neu, um die neue Version zu laden.`,
    summaryFailed: (ok: number, failed: number) =>
      `${ok} aktualisiert, ${failed} fehlgeschlagen. Fahre über einen Fehler für den Grund.`,
    summaryDeclined: (ok: number, declined: number) =>
      `${ok} aktualisiert. ${declined} nicht angewendet, da die neuen Berechtigungen abgelehnt wurden.`,
    thisProfile: '· dieses Profil',
    failed: 'Fehlgeschlagen',
    declined: 'Nicht angewendet',
    skippedOffline: 'Offline · übersprungen',
    skippedSignIn: 'Abgemeldet · übersprungen',
    skippedIdle: 'Nicht verbunden · übersprungen',
    skippedUnreadable: 'Nicht prüfbar',
    confirm: (installs: number) => `${installs} aktualisieren`,
    updating: 'Wird aktualisiert…',
    done: 'Fertig',
    consentBody: (names: string, installs: number) =>
      `Die neue Katalogversion von ${names} fügt Funktionen hinzu, die die installierte Version nicht hat. Die Zustimmung gilt für ${installs === 1 ? '1 Profil' : `${installs} Profile`}:`,
    finished: (installs: number) =>
      `${installs === 1 ? '1 Profil' : `${installs} Profile`} aktualisiert. Starte diese Gateways neu, um sie anzuwenden.`
  }
}
