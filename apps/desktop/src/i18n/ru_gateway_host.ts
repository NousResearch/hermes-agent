import type { GatewayHostTranslations } from './types_gateway_host'

// Settings → Gateways copy where the connection cannot change, spread into ru.ts.
export const ruGatewayHost: GatewayHostTranslations = {
  unavailableTitle: 'Настройки шлюза недоступны',
  unavailableDesc: 'IPC-мост приложения не предоставляет настройки шлюза.',
  webappHostTitle: 'Хост Hermes',
  webappHostDesc:
    'Webapp всегда использует хост Hermes, который его обслуживает. Чтобы сменить шлюз, войти в Hermes Cloud или управлять сохранёнными подключениями, используйте приложение Hermes Desktop.'
}
