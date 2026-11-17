import type { WebhookTestCopy } from './types_webhook_test'

export const zhWebhookTest: Partial<WebhookTestCopy> = {
  title: '测试此 Webhook',
  hint: '粘贴到终端即可发送一条已签名的事件。每次运行都有独立的投递 ID。',
  placeholderHint: '请将占位符替换为创建订阅时显示的密钥，或使用“发送测试”。',
  copyCommand: '复制命令',
  send: '发送测试',
  sending: '发送中…',
  accepted: (status, deliveryId) => `测试已投递（${status}）· ${deliveryId}`,
  rejected: (status, detail) => `接收端返回 ${status}：${detail}`,
  failed: '无法发送测试'
}
