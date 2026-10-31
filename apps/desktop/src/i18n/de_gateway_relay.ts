import type { GatewayRelayTranslations } from './types_gateway_relay'

// Settings → Gateways → Other gateways: `settings.gateway.relay` in de.ts.
export const deGatewayRelay: GatewayRelayTranslations = {
  title: 'Andere Gateways',
  intro:
    'Öffne ein anderes Hermes-Gateway aus dieser Webapp. Seine Profile und Chats erscheinen im Gateway-Umschalter; Anfragen laufen über dieses Gateway, signiert mit seinem eigenen Schlüssel, und handeln als das Konto, das du dort bestätigst.',
  empty: 'Noch keine anderen Gateways.',
  addTitle: 'Gateway verbinden',
  urlLabel: 'Webapp-Adresse',
  labelLabel: 'Name',
  connect: 'Verbinden',
  connecting: 'Anderes Gateway wird geöffnet…',
  connectFailed: 'Die Verbindung zu diesem Gateway konnte nicht gestartet werden',
  connected: label => `${label} verbunden`,
  denied: 'Das andere Gateway wurde nicht verbunden: Die Anfrage wurde dort abgelehnt.',
  failed: detail => `Das andere Gateway wurde nicht verbunden: ${detail}`,
  remove: 'Entfernen',
  removeConfirm: label =>
    `${label} entfernen? Dieses Gateway erreicht es dann nicht mehr, und seine Freigabe dort wird widerrufen.`,
  removeFailed: 'Dieses Gateway konnte nicht entfernt werden',
  test: 'Testen',
  reachable: 'Erreichbar',
  unreachable: error => `Nicht erreichbar: ${error}`,
  scopes: scopes => `Zusätzlich erlaubt: ${scopes}`,
  coreOnly: 'Nur Chats, Profile, Sitzungen und Einstellungen',
  grantsTitle: 'Gateways, die dieses erreichen',
  grantsIntro:
    'Gateways, denen du erlaubt hast, deine Webapp hierher weiterzuleiten. Wer dort mit deinem Konto angemeldet ist, kann diesen Rechner darüber nutzen.',
  grantsEmpty: 'Kein anderes Gateway erreicht dieses.',
  grantFrom: (origin, when) => `${origin} · erlaubt ${when}`,
  revoke: 'Widerrufen',
  revokeConfirm: name =>
    `${name} widerrufen? Es kann dieses Gateway nicht mehr erreichen, und seine offenen Verbindungen werden geschlossen.`,
  revokeFailed: 'Dieses Gateway konnte nicht widerrufen werden',
  loadFailed: 'Die Gateway-Weiterleitung konnte nicht geladen werden'
}
