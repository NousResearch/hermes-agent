import type { GatewayRelayTranslations } from './types_gateway_relay'

// Settings → Gateways → Other gateways: `settings.gateway.relay` in zh.ts.
export const zhGatewayRelay: GatewayRelayTranslations = {
  title: '其他网关',
  intro:
    '在此 Webapp 中打开另一个 Hermes 网关。它的配置文件和聊天会加入网关切换器；请求经由此网关转发，用此网关自己的密钥签名，并以你在对方那里批准的账户身份执行。',
  empty: '还没有其他网关。',
  addTitle: '连接网关',
  urlLabel: 'Webapp 地址',
  labelLabel: '名称',
  connect: '连接',
  connecting: '正在打开另一个网关…',
  connectFailed: '无法开始连接该网关',
  connected: label => `已连接 ${label}`,
  denied: '另一个网关未连接：请求在对方那里被拒绝。',
  failed: detail => `另一个网关未连接：${detail}`,
  remove: '移除',
  removeConfirm: label => `移除 ${label}？此网关将无法再访问它，它在对方那里的授权也会被撤销。`,
  removeFailed: '无法移除该网关',
  removeHere: '仅在此处移除',
  removeUnconfirmed: label => `仅在此处移除 ${label}？它在对方那里的授权仍然有效，直到你在 ${label} 上撤销。`,
  test: '测试',
  reachable: '可访问',
  unreachable: error => `无法访问：${error}`,
  scopes: scopes => `另外允许：${scopes}`,
  coreOnly: '仅限聊天、配置文件、会话和设置',
  grantsTitle: '访问此网关的网关',
  grantsIntro: '你允许将 Webapp 中继到这里的网关。任何以你的账户登录其中一个网关的人，都可以通过它使用这台机器。',
  grantsEmpty: '没有其他网关访问此网关。',
  grantFrom: (origin, when) => `${origin} · ${when}授权`,
  revoke: '撤销',
  revokeConfirm: name => `撤销 ${name}？它将无法再访问此网关，其打开的连接也会关闭。`,
  revokeFailed: '无法撤销该网关',
  loadFailed: '无法加载网关中继'
}
