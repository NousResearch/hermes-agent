import type { TranslationOverrides } from './define-locale'

/** German copy for the Skills ▸ Plugins tab; `skills.plugins`, composed by de.ts. */
export const deSkillsPlugins: NonNullable<NonNullable<TranslationOverrides['skills']>['plugins']> = {
  agentTitle: 'Agent-Plugins',
  agentBlurb:
    'Erweitert den Agenten für das gewählte Profil — Tools, Hooks, Anbieter. Wirkt nach einem Gateway-Neustart.',
  pageBlurb: 'Ein Plugin kann diese App, den Agent oder beides erweitern – jede Hälfte hat einen eigenen Schalter.',
  halfDesktop: 'Desktop',
  halfDesktopHint: 'diese App, gleich für jedes Profil',
  halfAgent: 'Agent',
  halfAgentIn: profile => `Agent in ${profile}`,
  defaultProfile: 'Hermes (Standard)',
  kindAgent: 'Agent',
  kindDesktop: 'Desktop',
  kindBoth: 'Agent + Desktop',
  installAgentHere: 'Hier installieren',
  installAgentHereTip: profile =>
    `Die Desktop-Hälfte ist in dieser App geladen, aber die Agent-Hälfte ist in ${profile} nicht installiert. Installieren Sie sie dort.`,
  installAgentHereNoOrigin:
    'Die Agent-Hälfte ist in diesem Profil nicht installiert, und dieses Paket wurde von Hand hineinkopiert (kein Katalogeintrag und kein Git-Remote), daher kann es nicht von hier aus installiert werden. Kopieren Sie seinen Ordner ins Profil oder installieren Sie es neu aus Git.',
  desktopHalfPending: 'wird kopiert…',
  desktopHalfPendingTip:
    'Dieses Paket enthält eine Desktop-Hälfte, die noch nicht in die App kopiert wurde. Verwenden Sie „Erneut scannen“ oder starten Sie die App neu.',
  desktopHalfRemote: 'nicht verfügbar (Remote-Backend)',
  desktopHalfRemoteTip:
    'Die Desktop-Hälfte dieses Pakets liegt auf der Festplatte des Remote-Backends, die diese App nicht lesen kann. Um sie hier zu nutzen, führen Sie „Aus Git installieren“ mit der Repository-URL des Pakets und aktiviertem Desktop-Ziel aus – dadurch wird die Desktop-Hälfte auf diesen Rechner geklont.',
  emptyAll: 'Noch keine Plugins.',
  empty: 'Für dieses Profil sind keine Agent-Plugins installiert.',
  emptyHint: 'Durchsuchen Sie unten den Katalog und installieren Sie ein geprüftes Plugin mit einem Klick.',
  loadFailed: 'Agent-Plugins konnten nicht geladen werden',
  toggleFailed: name => `${name} konnte nicht umgeschaltet werden`,
  toolsetOn: (name: string, profile: string) => `${name}-Agent-Tools für ${profile} aktiviert`,
  toolsetOff: (name: string, profile: string) => `${name}-Agent-Tools für ${profile} deaktiviert`,
  toolsetToggleFailed: (name: string) =>
    `Die ${name}-Agent-Tools konnten nicht umgeschaltet werden; das Desktop-Panel bleibt unverändert`,
  legacyBackend:
    'Dieses Backend ist älter als schlüsseladressierte Plugin-Schalter — aktualisieren Sie Hermes, um es hier zu verwalten.',
  portableBadge: 'tragbar',
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
  catalogTitle: 'Plugin-Katalog',
  catalogBrowse: 'Durchsuchen',
  catalogHide: 'Katalog-Browser ausblenden',
  catalogHint:
    'Klicken Sie bei einem Plugin auf „+ Zu diesem Agenten hinzufügen“ – geprüfte Einträge werden an ihrem gepinnten Commit in das gewählte Profil installiert. Gebündelte Agent+Desktop-Plugins bieten beide Hälften an.',
  alreadyInstalled: name => `${name} ist in diesem Profil bereits installiert.`,
  catalogProvenance: sha => `Aus dem Hermes-Katalog installiert${sha ? ` am Pin ${sha}` : ''}.`,
  pinnedProvenance: sha =>
    `An Commit ${sha} gepinnt. Updates werden abgelehnt, bis es mit einem neuen Pin neu installiert wird.`,
  pinnedBadge: sha => `gepinnt @ ${sha}`,
  update: 'Aktualisieren',
  updateToPin: sha => `Auf ${sha} aktualisieren`,
  updateFailed: name => `${name} konnte nicht aktualisiert werden`,
  updated: (name: string) => `${name} wurde aktualisiert. Starten Sie das Gateway neu, damit es wirksam wird.`,
  upToDate: (name: string) => `${name} ist bereits aktuell.`,
  updateConsentTitle: (name: string) => `${name} braucht Ihre Zustimmung`,
  updateConsentBody: (name: string) =>
    `Das Update von ${name} bringt Änderungen, die Sie prüfen sollten. Wenden Sie es nur an, wenn Sie der Quelle vertrauen:`,
  updateConsentConfirm: 'Update anwenden',
  uninstall: 'Deinstallieren',
  uninstallTip: (name: string, profile: string) => `${name} aus ${profile} deinstallieren`,
  uninstallConfirmTitle: (name: string) => `${name} deinstallieren?`,
  uninstallConfirmBody: (name: string, profile: string) =>
    `Dadurch werden die Dateien des Plugins aus dem Profil ${profile} gelöscht. Eine mitgelieferte Desktop-Hälfte wird ebenfalls entfernt. Sie können es jederzeit aus dem Katalog oder aus Git neu installieren.`,
  uninstallFailed: (name: string) => `${name} konnte nicht deinstalliert werden`,
  uninstalled: (name: string) => `${name} deinstalliert. Starten Sie das Gateway neu, um es zu entladen.`,
  uninstallDesktopTip: (name: string) => `${name} aus dieser App deinstallieren`,
  uninstallDesktopConfirmBody: (name: string) =>
    `Dadurch wird ${name} aus dem Ordner desktop-plugins auf diesem Computer gelöscht und sofort entladen. Sie können es jederzeit aus Git neu installieren oder den Ordner zurücklegen.`,
  uninstalledDesktop: (name: string) => `${name} deinstalliert.`,
  deepLinkErrorTitle: 'Plugin-Installationslink abgelehnt',
  deepLinkCatalogInvalidName: 'Der Katalogname im Link fehlt oder ist ungültig.',
  deepLinkCatalogUnknown: (name: string) =>
    `\u201E${name}\u201C ist nicht im Hermes-Plugin-Katalog. Es wurde nichts installiert.`,
  deepLinkCatalogUnavailable:
    'Der Hermes-Plugin-Katalog konnte nicht geladen werden. Prüfen Sie Ihre Verbindung und öffnen Sie den Link erneut.',
  settingsToggle: (name: string) => `Einstellungen: ${name}`,
  settingsForm: {
    save: 'Einstellungen speichern',
    saved: (name: string) => `Einstellungen von ${name} gespeichert.`,
    saveFailed: (name: string) => `Einstellungen von ${name} konnten nicht gespeichert werden`,
    required: 'Erforderlich',
    secretSet: '•••••••• (gesetzt)',
    secretStoredAs: (env: string) =>
      `Wird in der .env des Profils als ${env} gespeichert, nie in config.yaml; leer lassen, um den aktuellen Wert zu behalten.`
  }
}
