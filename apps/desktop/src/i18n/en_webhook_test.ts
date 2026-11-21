import type { WebhookTestCopy } from './types_webhook_test'

export const enWebhookTest: WebhookTestCopy = {
  title: 'Test this webhook',
  hint: 'Paste into a terminal to send one signed event. Each run gets its own delivery id.',
  placeholderHint: 'Replace the placeholder with the secret shown when this subscription was created, or use Send test.',
  copyCommand: 'Copy command',
  send: 'Send test',
  sending: 'Sending…',
  accepted: (status, deliveryId) => `Test delivered (${status}) · ${deliveryId}`,
  rejected: (status, detail) => `Receiver answered ${status}: ${detail}`,
  ignored: 'Delivered, but the subscription ignored it (its event or filter rules drop a test event)',
  failed: 'Test could not be sent'
}
