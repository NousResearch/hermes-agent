import type { TranslationOverride } from '@hermes/shared/i18n'

import type { Translations } from './types'

// The in-app browser pane's copy (`preview.web`), composed by fr.ts.
export const frPreviewWeb: TranslationOverride<Translations['preview']['web']> = {
  embeddedPreviewHint:
    "Certains sites bloquent les aperçus intégrés. Ouvrez la page d'origine dans un onglet du navigateur.",
  appFailedToBoot: "Échec du démarrage de l'application d'aperçu",
  serverNotFound: 'Serveur non trouvé',
  remoteLoopback:
    "Cette adresse pointe vers la machine qui exécute votre agent, pas vers celle-ci. Le panneau du navigateur charge les pages localement ; un serveur de développement distant nécessite donc une redirection de port ou un nom d'hôte accessible.",
  failedToLoad: "Échec du chargement de l'aperçu",
  tryAgain: 'Réessayer',
  restarting: 'Hermes redémarre...',
  askRestart: 'Demander à Hermes de redémarrer le serveur',
  lookingRestart: taskId => `Hermes recherche un serveur d'aperçu à redémarrer (${taskId})`,
  restartingTitle: "Redémarrage du serveur d'aperçu",
  restartingMessage: "Hermes travaille en arrière-plan. Surveillez la console d'aperçu pour suivre la progression.",
  startRestartFailed: message => `Impossible de démarrer le redémarrage du serveur : ${message}`,
  restartFailed: 'Échec du redémarrage du serveur',
  hideConsole: "Masquer la console d'aperçu",
  showConsole: "Afficher la console d'aperçu",
  hideDevTools: "Masquer les outils de développement d'aperçu",
  openDevTools: "Ouvrir les outils de développement d'aperçu",
  goBack: 'Retour',
  goForward: 'Suivant',
  reload: 'Recharger la page',
  address: 'Adresse',
  addressPlaceholder: 'Saisir une adresse',
  blankPageBody: "Saisissez une adresse ci-dessus pour naviguer, ou demandez à Hermes d'ouvrir une page.",
  finishedRestarting: message => `Hermes a terminé le redémarrage du serveur d'aperçu${message ? `: ${message}` : ''}`,
  failedRestarting: message => `Échec du redémarrage du serveur : ${message}`,
  unknownError: 'erreur inconnue',
  restartedTitle: "Serveur d'aperçu redémarré",
  reloadingNow: "Rechargement de l'aperçu maintenant.",
  restartFailedTitle: "Échec du redémarrage de l'aperçu",
  restartFailedMessage: "Hermes n'a pas pu redémarrer le serveur.",
  stillWorking:
    "Hermes travaille toujours, mais aucun résultat de redémarrage n'est arrivé. La commande du serveur peut être en cours d'exécution au premier plan.",
  workspaceReloading: "Espace de travail modifié, rechargement de l'aperçu",
  fileChanged: url => `Fichier modifié, rechargement de l'aperçu : ${url}`,
  filesChanged: (count, url) => `${count} modifications de fichier, rechargement de l'aperçu : ${url}`,
  watchFailed: message => `Impossible de surveiller le fichier d'aperçu : ${message}`,
  moduleMimeDescription:
    "Les scripts de module sont servis avec le mauvais type MIME. Cela signifie généralement qu'un serveur de fichiers statiques sert une application Vite/React au lieu du serveur de développement du projet.",
  loadFailedConsole: (code, message) => `Échec du chargement${code ? ` (${code})` : ''} : ${message}`,
  unreachableDescription: "La page d'aperçu n'a pas pu être atteinte.",
  openTarget: url => `Ouvrir ${url}`,
  fallbackTitle: 'Aperçu',
  annotate: 'Annoter',
  annotateOn: "Arrêter l'annotation",
  annotateNeedPage: "Ouvrez d'abord une page dans le navigateur intégré.",
  annotateFailed: "Impossible de démarrer le mode d'annotation",
  commenting: 'Ajout de commentaires',
  addComments: count => (count === 1 ? 'Ajouter 1 commentaire' : `Ajouter ${count} commentaires`),
  commentPlaceholder: 'Ajouter un commentaire...',
  commentTitle: n => `Commentaire ${n}`,
  saveComment: 'Enregistrer',
  cancelComment: 'Annuler le commentaire'
}
