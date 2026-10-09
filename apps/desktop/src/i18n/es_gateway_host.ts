import type { GatewayHostTranslations } from './types_gateway_host'

// Settings → Gateways copy where the connection cannot change, spread into es.ts.
export const esGatewayHost: GatewayHostTranslations = {
  unavailableTitle: 'Ajustes del gateway no disponibles',
  unavailableDesc:
    'Los ajustes de conexión solo se pueden cambiar desde la app Hermes Desktop en el equipo que la ejecuta.',
  webappHostTitle: 'Host de Hermes',
  webappHostDesc:
    'La Webapp siempre usa el host de Hermes que la sirve. Para cambiar de gateway, iniciar sesión en Hermes Cloud o gestionar conexiones guardadas, usa la app Hermes Desktop.'
}
