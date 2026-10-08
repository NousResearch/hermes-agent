import type { MemoryDiscoveryTranslations } from './types_memory_discovery'

export const deMemoryDiscovery = {
  installed: 'Installiert',
  availableToInstall: 'Zum Installieren verfügbar',
  installationRequired: 'Installation erforderlich',
  reviewInstall: 'Prüfen & installieren',
  exploreAll: 'Alle entdecken…',
  missing: 'Fehlend',
  installConsent:
    'Installiert und aktiviert das Plugin mit seinen Abhängigkeiten. Der aktive Speicheranbieter bleibt bis zur ausdrücklichen Auswahl unverändert.',
  builtin: 'Integriert',
  configureElsewhere:
    'Konfiguriere den Anbieter über die CLI oder aktualisiere Hermes für Einstellungen ohne Aktivierung.',
  notReady:
    'Schließe die Konfiguration ab und installiere fehlende Abhängigkeiten. Starte nach einer Installation das Backend neu und versuche es erneut.',
  useFailed: 'Anbieter konnte nicht aktiviert werden. Prüfe die Konfiguration und versuche es erneut.',

  activeProvider: name => `Aktiv: ${name}`,
  useProvider: 'Anbieter verwenden',
  loadFailed: 'Speicheranbieter konnten nicht geladen werden',
  ownerChanged:
    'Wechsle zur Verbindung und zum Profil zurück, in denen du diese Installation geöffnet hast, und versuche es erneut.',
  notDiscovered:
    'Das Paket wurde installiert, sein Speicheranbieter aber noch nicht erkannt. Kehre zu den Speichereinstellungen zurück, um erneut zu suchen.',
  installedNotice:
    'Anbieter erkannt. Konfiguriere ihn zuerst und wähle ihn anschließend ausdrücklich zur Verwendung aus.',
  backToMemory: 'Zurück zu den Speichereinstellungen',
  connect: 'Verbinden',
  reconnect: 'Neu verbinden',
  connectOAuth: 'Über OAuth verbinden',
  apiKeySet: 'API-Schlüssel gesetzt',
  oauthSet: 'OAuth verbunden',
  waitingConsent: 'Warte auf Zustimmung im Browser…',
  stopWaiting: 'Nicht mehr warten',
  stoppedWaiting: 'Warten beendet. Die Autorisierung kann noch ausstehen.',
  startFailed: 'Verbindung konnte nicht gestartet werden.',
  connectionFailed: 'Verbindung fehlgeschlagen.'
} satisfies Partial<MemoryDiscoveryTranslations>
