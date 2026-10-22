import type { MessagingGatewayTranslations } from './types_messaging_gateway'

export const jaMessagingGateway = {
  hintGatewayStopped: 'ここからメッセージングゲートウェイを起動して接続してください。',
  restartNeeded: '保存しました。新しい設定を反映するにはメッセージングゲートウェイを再起動してください。',
  restartNow: '今すぐ再起動',
  restarting: '再起動中…',
  restartFailedManual: 'ゲートウェイの再起動に失敗しました。手動で再起動し、ゲートウェイのログを確認してください。',
  restartFailedManualDetail:
    '再起動をもう一度お試しください。失敗が続く場合は、ログを開いて診断情報を送信してください。',
  startMessagingGateway: 'メッセージングゲートウェイを起動',
  startingMessagingGateway: 'メッセージングゲートウェイを起動中…',
  gatewayStartFailed: 'メッセージングゲートウェイの起動に失敗しました。'
} satisfies MessagingGatewayTranslations
