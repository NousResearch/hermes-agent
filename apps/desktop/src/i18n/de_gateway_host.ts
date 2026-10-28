import type { GatewayHostTranslations } from './types_gateway_host'

// Settings → Gateways copy where the connection cannot change, spread into de.ts.
export const deGatewayHost: GatewayHostTranslations = {
  unavailableTitle: 'Gateway-Einstellungen nicht verfügbar',
  unavailableDesc: 'Die Desktop-IPC-Brücke stellt keine Gateway-Einstellungen bereit.',
  webappHostTitle: 'Hermes-Host',
  webappHostDesc:
    'Die Webapp nutzt immer den Hermes-Host, der sie ausliefert. Um das Gateway zu wechseln, sich bei Hermes Cloud anzumelden oder gespeicherte Verbindungen zu verwalten, verwenden Sie die Hermes-Desktop-App.',
  webAccess: {
    title: 'Webzugriff',
    intro:
      'Öffne Hermes auf diesem Computer vom Handy oder einem anderen Computer aus, angemeldet mit deinem Nous-Konto. Wer sich anmeldet, kann Befehle auf diesem Computer ausführen.',
    failed: 'Der Webzugriff konnte nicht geändert werden',
    running: 'Läuft',
    stopped: 'Gestoppt',
    starting: 'Wird gestartet',
    stopping: 'Wird gestoppt',
    restartNeeded: 'Neustart nötig',
    unavailable: 'Nicht verfügbar',
    notRunningDetail: 'Starte ihn, um Hermes von einem anderen Gerät aus zu öffnen.',
    needsNous: 'Melde dich zuerst unter Anbieter bei deinem Nous-Konto an.',
    runningLan: 'Öffne diese Adresse auf einem Gerät im selben Netzwerk.',
    runningPublic: 'Über deine URL bereitgestellt.',
    runningLocal:
      'Nur dieser Computer kann die Web-App öffnen. Stoppe sie und starte sie erneut, um sie von anderen Geräten aus zu öffnen.',
    pendingOn: 'Gespeichert. Das Gateway stellt die Web-App nach seinem Neustart bereit.',
    pendingOff: 'Ausgeschaltet, aber das Gateway stellt die Web-App bis zu seinem Neustart weiter bereit.',
    unavailableDetail: reason => `Das Gateway konnte die Web-App nicht bereitstellen: ${reason}`,
    restartBusy:
      'Gespeichert. Ein Chat läuft, deshalb wurde das Gateway nicht neu gestartet. Wähle „Gateway neu starten“, sobald er beendet ist.',
    restartFailed:
      'Gespeichert, aber das Gateway konnte nicht neu starten. Wähle „Gateway neu starten“, um es erneut zu versuchen.',
    startTimedOut: 'Das Gateway hat die Web-App nicht innerhalb von 90 Sekunden bereitgestellt.',
    stopTimedOut: 'Das Gateway hat die Bereitstellung der Web-App nicht innerhalb von 90 Sekunden beendet.',
    start: 'Starten',
    stop: 'Stoppen',
    restartGateway: 'Gateway neu starten',
    modeLan: 'Dieses Netzwerk',
    modeLanDetail: url => `Geräte in deinem WLAN öffnen ${url}. Nur in einem vertrauenswürdigen Netzwerk verwenden.`,
    lanUnencrypted:
      'Die Verbindung ist nicht verschlüsselt: Andere in diesem Netzwerk könnten mitlesen, was du sendest und empfängst.',
    noLan: 'Dieser Computer hat keine Netzwerkadresse, die andere Geräte erreichen können.',
    modePublic: 'Eigene URL',
    modePublicDetail: port =>
      `Eine HTTPS-Adresse, die du bereits betreibst, etwa Tailscale Serve oder ein Tunnel, die an http://127.0.0.1:${port} weiterleitet.`
  }
}
