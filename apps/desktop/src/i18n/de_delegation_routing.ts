import type { DelegationRoutingTranslations } from './types_delegation_routing'

export const deDelegationRouting: DelegationRoutingTranslations = {
  title: 'Routing',
  settingsTitle: 'Subagent-Standards',
  intro: 'Wie jeder Subagent sein Modell erhält.',
  modes: {
    default: { label: 'Standard', description: 'Jeder Subagent läuft mit dem Subagent-Modell unten.' },
    model_map: {
      label: 'Modellrouten',
      description: 'Benennen Sie einige Modelle. Der Agent wählt für jeden Subagenten eines aus.'
    },
    profile_routes: {
      label: 'Profilrouten',
      description: 'Subagenten laufen als eines Ihrer Profile, mit dessen eigenem Modell, Skills und Schlüsseln.'
    }
  },
  inUse: 'Aktiv',
  plugin: 'Plugin',
  pluginMissing: 'Modellrouten und Profilrouten stammen aus dem Plugin Advanced Delegation.',
  pluginDisabled: 'Advanced Delegation ist installiert, aber für dieses Profil ausgeschaltet.',
  pluginNotLoaded:
    'Advanced Delegation ist eingeschaltet, wurde von Hermes aber noch nicht geladen. Starten Sie Hermes neu, um den Vorgang abzuschließen.',
  installPlugin: 'Plugin installieren',
  installUnavailable:
    'Advanced Delegation ist noch nicht im Plugin-Katalog. Installieren Sie es über Fähigkeiten › Plugins.',
  enablePlugin: 'Einschalten',
  enableFailed: 'Advanced Delegation konnte nicht eingeschaltet werden',
  modeFailed: 'Subagent-Routing konnte nicht geändert werden',
  loadFailed: 'Subagent-Routing konnte nicht geladen werden',
  blockedPill: 'Läuft nicht',
  blockedDetail:
    'Dieser Modus ist gespeichert, daher lehnt Hermes jeden Subagenten ab, statt stillschweigend Standard zu verwenden.',
  blockedMissing: 'Subagenten starten erst, wenn Advanced Delegation installiert ist.',
  blockedDisabled: 'Subagenten starten erst, wenn Advanced Delegation für dieses Profil eingeschaltet ist.',
  blockedNotLoaded:
    'Subagenten starten erst, wenn Hermes Advanced Delegation geladen hat. Starten Sie Hermes neu, um den Vorgang abzuschließen.',
  blockedUnavailable: label =>
    `Subagenten starten erst, wenn ${label} hier laufen kann. Sein Plugin meldet, dass es das nicht kann.`,
  blockedUnregistered: name =>
    `Subagenten starten erst, wenn ein Plugin, das „${name}“ bereitstellt, installiert und eingeschaltet ist.`,
  thirdPartyDescription: 'Stammt aus einem Plugin. Seine Einstellungen liegen bei diesem Plugin.',

  modelRoutesTitle: 'Modellrouten',
  modelRoutesEmpty: 'Noch keine Routen. Bis Sie eine hinzufügen, verwenden Subagenten das Modell des Hauptagenten.',
  addRoute: 'Route hinzufügen',
  editRoute: 'Bearbeiten',
  makeDefault: 'Als Standard festlegen',
  deleteRoute: 'Löschen',
  defaultBadge: 'Standard',
  routeActions: name => `Aktionen für ${name}`,
  deleteConfirmTitle: name => `Route „${name}“ löschen?`,
  deleteConfirmBody: 'Subagenten können sie dann nicht mehr wählen.',
  deleteConfirmNextDefault: name => `Sie ist die Standardroute, daher wird „${name}“ zum Standard.`,
  deleteConfirmAction: 'Route löschen',
  saveFailed: 'Routen konnten nicht gespeichert werden',
  baseUrlRoute: url => `Endpunkt ${url}`,

  dialogNew: 'Neue Modellroute',
  dialogEdit: 'Modellroute bearbeiten',
  dialogDesc: 'Der Agent liest Name und Beschreibung, um zu entscheiden, welche Route ein Subagent nimmt.',
  nameLabel: 'Name',
  namePlaceholder: 'z. B. fast',
  nameHint: 'Kurz, ohne Leerzeichen. Der Agent wählt Routen anhand dieses Namens.',
  descriptionLabel: 'Wann sie genutzt wird',
  descriptionPlaceholder: 'z. B. Günstig und schnell: Nachschlagen, Zusammenfassungen, einfache Änderungen.',
  descriptionHint: 'Eine Zeile, die der Agent vor der Wahl liest.',
  providerLabel: 'Anbieter',
  providerPlaceholder: 'Anbieter wählen',
  modelLabel: 'Modell',
  effortLabel: 'Denkanstrengung',
  effortInherit: 'Subagent-Standard',
  tierLabel: 'Geschwindigkeit',
  tierNormal: 'Normal',
  tierFast: 'Schnell',
  tierUltrafast: 'Ultraschnell',
  tierHint: 'Nur bei Modellen, die das anbieten.',
  optional: 'optional',
  errors: {
    required: 'Erforderlich',
    spaces: 'Keine Leerzeichen',
    duplicate: 'Eine andere Route verwendet diesen Namen bereits'
  },
  saveRoute: 'Route speichern',
  createRoute: 'Route hinzufügen',

  profileRoutesTitle: 'Profile, als die Subagenten laufen können',
  profileRoutesHint: 'Jedes ausgewählte Profil bringt sein eigenes Modell, Skills, Persona und Schlüssel mit.',
  profileDescriptionPlaceholder: 'Wann es genutzt wird (optional)',
  profileDefault: 'Standard',
  profileEmpty: 'Noch keine weiteren Profile. Erstellen Sie eines über die Profil-Leiste.',
  profileMissing: name => `${name} (Profil existiert nicht mehr)`,

  inactiveTitle: 'In diesem Modus nicht verwendet',
  inactiveModelMap: route =>
    `Modellrouten wählen das Modell jedes Subagenten. Subagenten ohne gewählte Route verwenden „${route}“.`,
  inactiveProfiles: profile =>
    `Profilrouten wählen das Profil jedes Subagenten. Subagenten ohne gewähltes Profil laufen als „${profile}“.`,
  inactiveModelMapNoDefault:
    'Modellrouten wählen das Modell jedes Subagenten. Subagenten ohne Route verwenden das Modell des Hauptagenten.',
  inactiveProfilesNoDefault:
    'Profilrouten wählen das Profil jedes Subagenten. Subagenten ohne Profil verwenden das Modell des Hauptagenten.',
  inactivePlugin: label => `${label} legt das Modell jedes Subagenten fest, daher gilt diese Einstellung nicht.`,
  useDefault: 'Zu Standard wechseln'
}
