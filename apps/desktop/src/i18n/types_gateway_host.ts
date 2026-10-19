// Settings → Gateways copy for a renderer that cannot change its connection: a Desktop whose
// bridge exposes no gateway settings, and the Webapp, bound to the host that serves it.
// `Translations['settings']['gateway']` spreads this in.
export interface GatewayHostTranslations {
  unavailableTitle: string
  unavailableDesc: string
  webappHostTitle: string
  webappHostDesc: string
}
