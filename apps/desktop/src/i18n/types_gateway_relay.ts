// Settings → Gateways copy for the gateway relay (hermes_cli/webapp_relay.py): the gateways the
// Webapp's own gateway reaches, and the gateways that reach this one. English only so far; other
// locales fall back to it. `Translations['settings']['gateway']['relay']`.
export interface GatewayRelayTranslations {
  title: string
  intro: string
  empty: string
  addTitle: string
  urlLabel: string
  labelLabel: string
  connect: string
  connecting: string
  connectFailed: string
  connected: (label: string) => string
  denied: string
  failed: (detail: string) => string
  remove: string
  removeConfirm: (label: string) => string
  removeFailed: string
  /** The confirm button that forgets a gateway here when it did not confirm the revocation. */
  removeHere: string
  removeUnconfirmed: (label: string) => string
  test: string
  reachable: string
  unreachable: (error: string) => string
  scopes: (scopes: string) => string
  coreOnly: string
  grantsTitle: string
  grantsIntro: string
  grantsEmpty: string
  grantFrom: (origin: string, when: string) => string
  revoke: string
  revokeConfirm: (name: string) => string
  revokeFailed: string
  loadFailed: string
}
