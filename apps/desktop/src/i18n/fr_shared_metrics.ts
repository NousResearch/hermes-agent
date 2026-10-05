
export const frSharedMetrics = {
  consentTitle: 'Aider à améliorer Hermes ?',
  consentBody:
    'Les métriques partagées ne contiennent que des compteurs bornés. Jamais de prompts, fichiers, chemins ni textes d’erreur. La collecte reste locale. Les envoyer à Nous est un consentement distinct.',
  whatIsCollected: 'Ce qui est collecté',
  collectedIntro: 'Uniquement des compteurs bornés :',
  collectedActivity: 'Activité, durée des sessions, résultats et classes d’erreur, y compris un motif issu d’une liste fixe quand une écriture en mémoire ou une compression du contexte est refusée, échoue ou est ignorée',
  collectedModels: 'Routes de modèles et totaux de tokens',
  collectedNames: 'Noms des outils, commandes et éléments du catalogue intégrés',
  collectedMilestones: 'Comptes de configuration regroupés',
  collectedReliability:
    'Résultats et durée des mises à jour, plantages, vitesse de démarrage et de réponse, état des plateformes de messagerie',
  collectedUsage:
    "Comment Hermes est utilisé : précision et efficacité de l'agent (modifications réussies, boucles, reprises après erreur, jetons et appels d'outils par tâche, ruptures de cache), temps actif par interface et mode Desktop, zones, actions et réglages de l'app utilisés, vite fermés ou désactivés, et résultats de la configuration des fournisseurs",
  collectedMachine:
    "Données générales de la machine : plage de RAM, type de GPU, âge et canal de la version de Hermes, mises à jour en retard, utilisation d'un serveur de modèles local",
  installId:
    'L’envoi transmet chaque paquet quotidien au service de télémétrie de Nous. Les paquets portent l’identifiant d’installation de ce profil : un UUID aléatoire stable sans information personnelle, réinitialisé en supprimant le dossier des métriques partagées.',
  consentWindow:
    'Seuls les paquets dont toute la période de collecte tombe dans une fenêtre de consentement enregistrée sont envoyés — les données d’avant votre accord, ou de toute période où l’envoi était désactivé, restent sur cette machine. L’envoi peut être désactivé à tout moment.',
  readDocs: 'Lire tous les détails',
  share: 'Collecter et envoyer à Nous',
  local: 'Collecter en local uniquement',
  off: 'Non merci',
  changeLater: 'Vous pouvez changer cela à tout moment dans Réglages → Sécurité.',
  saveFailed: 'Impossible d’enregistrer votre choix',
  collectLabel: 'Collecter les statistiques d’utilisation',
  collectDesc:
    'Compteurs bornés conservés sur cet appareil. Jamais de prompts, fichiers, chemins ni textes d’erreur.',
  sendLabel: 'Envoyer les statistiques d’utilisation à Nous',
  sendDesc:
    'Envoyer chaque paquet quotidien au service de télémétrie de Nous. Seules les données d’une fenêtre de consentement sont envoyées. Nécessite la collecte activée.',
  unavailable: 'Mettez à jour le backend Hermes pour modifier ce réglage.',
  stripBody: 'Uniquement des compteurs bornés, jamais de prompts ni de fichiers.',
  stripReaskBody: 'Nouvelle demande : une version précédente pouvait enregistrer « Non merci » avant que vous ne voyiez cette question.',
  stripChoices: { share: 'Envoyer à Nous', local: 'Local uniquement', off: 'Non merci' },
  stripDetails: 'Détails'
}
