import type { GatewayRelayTranslations } from './types_gateway_relay'

// Settings → Gateways → Other gateways: `settings.gateway.relay` in zh-hant_settings.ts.
export const zhHantGatewayRelay: GatewayRelayTranslations = {
  title: '其他閘道',
  intro:
    '在此 Webapp 中開啟另一個 Hermes 閘道。它的設定檔與聊天會加入閘道切換器；請求經由此閘道轉送，以此閘道自己的金鑰簽署，並以你在對方那裡核准的帳戶身分執行。',
  empty: '尚無其他閘道。',
  addTitle: '連線閘道',
  urlLabel: 'Webapp 位址',
  labelLabel: '名稱',
  connect: '連線',
  connecting: '正在開啟另一個閘道…',
  connectFailed: '無法開始連線該閘道',
  connected: label => `已連線 ${label}`,
  denied: '另一個閘道未連線：請求在對方那裡遭到拒絕。',
  failed: detail => `另一個閘道未連線：${detail}`,
  remove: '移除',
  removeConfirm: label => `移除 ${label}？此閘道將無法再存取它，它在對方那裡的授權也會被撤銷。`,
  removeFailed: '無法移除該閘道',
  removeHere: '僅在此處移除',
  removeUnconfirmed: label => `僅在此處移除 ${label}？它在對方那裡的授權仍然有效，直到你在 ${label} 上撤銷。`,
  test: '測試',
  reachable: '可存取',
  unreachable: error => `無法存取：${error}`,
  scopes: scopes => `另外允許：${scopes}`,
  coreOnly: '僅限聊天、設定檔、工作階段與設定',
  grantsTitle: '存取此閘道的閘道',
  grantsIntro: '你允許將 Webapp 中繼到這裡的閘道。任何以你的帳戶登入其中一個閘道的人，都可以透過它使用這台電腦。',
  grantsEmpty: '沒有其他閘道存取此閘道。',
  grantFrom: (origin, when) => `${origin} · ${when}授權`,
  revoke: '撤銷',
  revokeConfirm: name => `撤銷 ${name}？它將無法再存取此閘道，其開啟的連線也會關閉。`,
  revokeFailed: '無法撤銷該閘道',
  loadFailed: '無法載入閘道中繼'
}
