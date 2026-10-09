import type { GatewayHostTranslations } from './types_gateway_host'

// Settings → Gateways copy where the connection cannot change, spread into en.ts.
export const enGatewayHost: GatewayHostTranslations = {
  unavailableTitle: 'Gateway settings unavailable',
  unavailableDesc: 'Connection settings can only be changed from the Hermes Desktop app on the computer running it.',
  webappHostTitle: 'Hermes host',
  webappHostDesc:
    'The Webapp always uses the Hermes host that serves it. To switch gateways, sign in to Hermes Cloud or manage saved connections, use the Hermes Desktop app.'
}
