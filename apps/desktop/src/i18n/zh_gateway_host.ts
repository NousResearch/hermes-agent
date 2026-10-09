import type { GatewayHostTranslations } from './types_gateway_host'

// Settings → Gateways copy where the connection cannot change, spread into zh.ts.
export const zhGatewayHost: GatewayHostTranslations = {
  unavailableTitle: '网关设置不可用',
  unavailableDesc: '桌面 IPC 桥未暴露网关设置。',
  webappHostTitle: 'Hermes 主机',
  webappHostDesc:
    'Webapp 始终使用提供它的 Hermes 主机。如需切换网关、登录 Hermes Cloud 或管理已保存的连接，请使用 Hermes Desktop 应用。'
}
