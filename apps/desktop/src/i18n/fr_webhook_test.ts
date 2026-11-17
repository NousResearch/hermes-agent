import type { WebhookTestCopy } from './types_webhook_test'

export const frWebhookTest: Partial<WebhookTestCopy> = {
  title: 'Tester ce webhook',
  hint: 'Collez dans un terminal pour envoyer un événement signé. Chaque exécution a son propre identifiant de livraison.',
  placeholderHint: "Remplacez l'espace réservé par le secret affiché à la création de l'abonnement, ou utilisez Envoyer un test.",
  copyCommand: 'Copier la commande',
  send: 'Envoyer un test',
  sending: 'Envoi…',
  accepted: (status, deliveryId) => `Test livré (${status}) · ${deliveryId}`,
  rejected: (status, detail) => `Le récepteur a répondu ${status} : ${detail}`,
  failed: "Le test n'a pas pu être envoyé"
}
