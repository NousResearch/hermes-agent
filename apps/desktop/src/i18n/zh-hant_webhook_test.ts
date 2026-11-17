import type { WebhookTestCopy } from './types_webhook_test'

export const zhHantWebhookTest: Partial<WebhookTestCopy> = {
  title: '測試此 Webhook',
  hint: '貼到終端機即可傳送一則已簽署的事件。每次執行都有獨立的投遞 ID。',
  placeholderHint: '請將預留位置替換為建立訂閱時顯示的密鑰，或使用「傳送測試」。',
  copyCommand: '複製指令',
  send: '傳送測試',
  sending: '傳送中…',
  accepted: (status, deliveryId) => `測試已投遞（${status}）· ${deliveryId}`,
  rejected: (status, detail) => `接收端回應 ${status}：${detail}`,
  failed: '無法傳送測試'
}
