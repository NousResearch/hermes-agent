import type { GatewayHostTranslations } from './types_gateway_host'

// Settings → Gateways copy where the connection cannot change, spread into ja.ts.
export const jaGatewayHost: GatewayHostTranslations = {
  unavailableTitle: 'ゲートウェイ設定は利用できません',
  unavailableDesc: 'デスクトップ IPC ブリッジはゲートウェイ設定を公開していません。',
  webappHostTitle: 'Hermes ホスト',
  webappHostDesc:
    'Webapp は常に配信元の Hermes ホストを使用します。ゲートウェイの切り替え、Hermes Cloud へのサインイン、保存済み接続の管理は Hermes Desktop アプリで行ってください。'
}
