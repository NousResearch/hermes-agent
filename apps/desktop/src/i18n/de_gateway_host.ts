import type { GatewayHostTranslations } from './types_gateway_host'

// Settings → Gateways copy where the connection cannot change, spread into de.ts.
export const deGatewayHost: GatewayHostTranslations = {
  unavailableTitle: 'Gateway-Einstellungen nicht verfügbar',
  unavailableDesc: 'Die Desktop-IPC-Brücke stellt keine Gateway-Einstellungen bereit.',
  webappHostTitle: 'Hermes-Host',
  webappHostDesc:
    'Die Webapp läuft immer auf dem Hermes-Host, der sie bereitstellt. Die gespeicherten Verbindungen unten sind die Liste dieses Rechners, geteilt mit der Hermes-Desktop-App; andere Gateways werden über diesen Host erreicht.',
  webappOauthPaste: {
    title: 'Anmeldung abschließen',
    description:
      'Wenn der Anmelde-Tab auf einer Adresse endete, die nicht lädt (dieser Browser läuft nicht auf dem Rechner des Gateways), füge diese Adresse hier ein.',
    placeholder: 'http://127.0.0.1:…/webapp/gateway-connections/oauth/callback?…',
    finish: 'Abschließen',
    failed: 'Anmeldung konnte nicht abgeschlossen werden'
  },
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
    restartFailedBecause: reason => `Das Gateway konnte nicht neu starten: ${reason}`,
    startTimedOut: 'Das Gateway hat die Web-App nicht innerhalb von 90 Sekunden bereitgestellt.',
    stopTimedOut: 'Das Gateway hat die Bereitstellung der Web-App nicht innerhalb von 90 Sekunden beendet.',
    startStages: {
      preparing: 'Die Web-App wird vorbereitet. Das Erstellen kann einige Minuten dauern.',
      building: 'Die Web-App wird erstellt. Das kann einige Minuten dauern.',
      registering: 'Die Nous-Anmeldung wird für diese Adresse eingerichtet.',
      restarting: 'Das Gateway wird neu gestartet, um die Web-App bereitzustellen.'
    },
    startElapsed: time => `Verstrichene Zeit: ${time}`,
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
