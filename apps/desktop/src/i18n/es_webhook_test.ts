import type { WebhookTestCopy } from './types_webhook_test'

export const esWebhookTest: Partial<WebhookTestCopy> = {
  title: 'Probar este webhook',
  hint: 'Pégalo en una terminal para enviar un evento firmado. Cada ejecución tiene su propio id de entrega.',
  placeholderHint: 'Sustituye el marcador por el secreto que se mostró al crear la suscripción, o usa Enviar prueba.',
  copyCommand: 'Copiar comando',
  send: 'Enviar prueba',
  sending: 'Enviando…',
  accepted: (status, deliveryId) => `Prueba entregada (${status}) · ${deliveryId}`,
  rejected: (status, detail) => `El receptor respondió ${status}: ${detail}`,
  failed: 'No se pudo enviar la prueba'
}
