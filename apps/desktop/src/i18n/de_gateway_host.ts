import type { GatewayHostTranslations } from './types_gateway_host'

// Settings → Gateways copy where the connection cannot change, spread into de.ts.
export const deGatewayHost: GatewayHostTranslations = {
  unavailableTitle: 'Gateway-Einstellungen nicht verfügbar',
  unavailableDesc: 'Die Desktop-IPC-Brücke stellt keine Gateway-Einstellungen bereit.',
  webappHostTitle: 'Hermes-Host',
  webappHostDesc:
    'Die Webapp nutzt immer den Hermes-Host, der sie ausliefert. Um das Gateway zu wechseln, sich bei Hermes Cloud anzumelden oder gespeicherte Verbindungen zu verwalten, verwenden Sie die Hermes-Desktop-App.'
}
