import type { WebhookTestCopy } from './types_webhook_test'

export const deWebhookTest: Partial<WebhookTestCopy> = {
  title: 'Diesen Webhook testen',
  hint: 'In ein Terminal einfügen, um ein signiertes Ereignis zu senden. Jeder Lauf erhält eine eigene Zustellungs-ID.',
  placeholderHint: 'Ersetzen Sie den Platzhalter durch das Secret, das beim Anlegen angezeigt wurde, oder nutzen Sie „Test senden“.',
  copyCommand: 'Befehl kopieren',
  send: 'Test senden',
  sending: 'Wird gesendet…',
  accepted: (status, deliveryId) => `Test zugestellt (${status}) · ${deliveryId}`,
  rejected: (status, detail) => `Empfänger antwortete ${status}: ${detail}`,
  ignored: 'Zugestellt, aber das Abonnement hat es ignoriert (seine Ereignis- oder Filterregeln verwerfen ein Testereignis)',
  failed: 'Test konnte nicht gesendet werden'
}
