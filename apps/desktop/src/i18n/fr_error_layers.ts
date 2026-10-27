import type { Translations } from './types'

type ThreadCopy = Translations['assistant']['thread']

// The error card's layer fallbacks (lib/error-surface-copy.ts): used when a failure code has no own entry.
export const frErrorLayers: Pick<ThreadCopy, 'errorLayerBodies' | 'errorLayers'> = {
  errorLayers: {
    auth: "Erreur d'authentification",
    billing: 'Crédits épuisés',
    disk: 'Disque plein',
    endpoint: 'Erreur du point de terminaison personnalisé',
    gateway: 'Erreur du gateway',
    generic: 'Échec du tour',
    provider: 'Erreur du fournisseur',
    runtime: "Erreur d'exécution locale",
    streaming: 'Erreur de connexion au flux'
  },
  errorLayerBodies: {
    auth: "Le service d'IA a refusé votre connexion. Vérifiez les identifiants de ce fournisseur, puis renvoyez votre message.",
    billing:
      "Votre compte n'a plus de crédits chez ce fournisseur. Rechargez-le ou changez de fournisseur, puis réessayez.",
    disk: "Votre disque est plein ; Hermes n'a pas pu enregistrer cette conversation. Libérez de l'espace, puis réessayez.",
    endpoint:
      "Hermes ne parvient pas à joindre votre serveur de modèle personnalisé. Vérifiez qu'il fonctionne, puis renvoyez votre message.",
    gateway:
      'Hermes a rencontré un problème interne au démarrage de cette réponse. Renvoyez votre message ; si cela persiste, envoyez les diagnostics.',
    generic:
      "Une erreur s'est produite pendant la réponse de Hermes. Réessayez ou copiez les détails si cela persiste.",
    provider: "Le service d'IA n'a pas pu traiter cette demande. Réessayez dans un instant ou changez de fournisseur.",
    runtime:
      'Hermes a rencontré un problème interne au démarrage de cette réponse. Renvoyez votre message ; si cela persiste, envoyez les diagnostics.',
    streaming: 'La connexion a été interrompue avant la fin de la réponse. Réessayez pour la renvoyer.'
  }
}
