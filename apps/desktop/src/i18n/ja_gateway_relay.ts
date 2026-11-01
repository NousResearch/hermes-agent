import type { GatewayRelayTranslations } from './types_gateway_relay'

// Settings → Gateways → Other gateways: `settings.gateway.relay` in ja.ts.
export const jaGatewayRelay: GatewayRelayTranslations = {
  title: 'ほかのゲートウェイ',
  intro:
    'この Webapp から別の Hermes ゲートウェイを開きます。そのプロファイルとチャットがゲートウェイ切り替えに加わります。リクエストはこのゲートウェイを経由し、このゲートウェイ自身の鍵で署名され、相手側で承認したアカウントとして動作します。',
  empty: 'ほかのゲートウェイはまだありません。',
  addTitle: 'ゲートウェイを接続',
  urlLabel: 'Webapp のアドレス',
  labelLabel: '名前',
  connect: '接続',
  connecting: '相手のゲートウェイを開いています…',
  connectFailed: 'そのゲートウェイへの接続を開始できませんでした',
  connected: label => `${label} を接続しました`,
  denied: '相手のゲートウェイは接続されませんでした：相手側でリクエストが拒否されました。',
  failed: detail => `相手のゲートウェイは接続されませんでした：${detail}`,
  remove: '削除',
  removeConfirm: label =>
    `${label} を削除しますか？ このゲートウェイからは届かなくなり、相手側の許可も取り消されます。`,
  removeFailed: 'そのゲートウェイを削除できませんでした',
  removeHere: 'ここだけ削除',
  removeUnconfirmed: label =>
    `${label} をここだけ削除しますか？ 相手側の許可は、${label} で取り消すまで有効なままです。`,
  test: 'テスト',
  reachable: '到達可能',
  unreachable: error => `到達できません：${error}`,
  scopes: scopes => `追加で許可：${scopes}`,
  coreOnly: 'チャット、プロファイル、セッション、設定のみ',
  grantsTitle: 'このゲートウェイに届くゲートウェイ',
  grantsIntro:
    'あなたの Webapp をここへ中継することを許可したゲートウェイです。そのいずれかにあなたのアカウントでサインインしている人は、それを通じてこのマシンを使えます。',
  grantsEmpty: 'このゲートウェイに届くほかのゲートウェイはありません。',
  grantFrom: (origin, when) => `${origin} · ${when}に許可`,
  revoke: '取り消す',
  revokeConfirm: name =>
    `${name} の許可を取り消しますか？ このゲートウェイに届かなくなり、開いている接続は閉じられます。`,
  revokeFailed: 'そのゲートウェイの許可を取り消せませんでした',
  loadFailed: 'ゲートウェイ中継を読み込めませんでした'
}
