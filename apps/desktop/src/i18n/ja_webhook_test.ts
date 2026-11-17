import type { WebhookTestCopy } from './types_webhook_test'

export const jaWebhookTest: Partial<WebhookTestCopy> = {
  title: 'この Webhook をテスト',
  hint: 'ターミナルに貼り付けると署名付きイベントを 1 件送信します。実行ごとに個別の配信 ID が付きます。',
  placeholderHint: 'プレースホルダーを作成時に表示されたシークレットに置き換えるか、「テスト送信」を使ってください。',
  copyCommand: 'コマンドをコピー',
  send: 'テスト送信',
  sending: '送信中…',
  accepted: (status, deliveryId) => `テストを配信しました（${status}）· ${deliveryId}`,
  rejected: (status, detail) => `受信側の応答 ${status}: ${detail}`,
  ignored: '配信されましたが、サブスクリプションが無視しました（イベントまたはフィルターのルールがテストイベントを除外しています）',
  failed: 'テストを送信できませんでした'
}
