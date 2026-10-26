// Notification settings copy used when the desktop locale loads.
export const frNotifications = {
  title: 'Notifications',
  intro: "Notifications système (pas les toasts de l'application). Par appareil.",
  enableAll: 'Activer les notifications',
  enableAllDesc: 'Désactivé coupe toutes les notifications ci-dessous.',
  focusedHint: 'Les alertes de fin ne se déclenchent que quand Hermes est en arrière-plan.',
  kinds: {
    approval: {
      label: 'Approbation requise',
      description: "Une commande attend que vous l'approuviez ou la rejetiez."
    },
    input: {
      label: 'Saisie requise',
      description: "Hermes a posé une question ou a besoin d'un mot de passe ou d'un secret."
    },
    turnDone: {
      label: 'Réponse prête',
      description: "Un tour s'est terminé pendant que Hermes était en arrière-plan."
    },
    turnError: {
      label: 'Échec du tour',
      description: 'Erreurs de tour en arrière-plan.'
    },
    backgroundDone: {
      label: 'Tâche en arrière-plan terminée',
      description: "Une commande terminal en arrière-plan s'est terminée."
    },
    credits: {
      label: 'Alertes de crédits',
      description: "L'accès aux crédits est suspendu ou rétabli."
    },
    plugin: {
      label: 'Notifications des plugins',
      description: 'Un plugin desktop a envoyé une notification pendant que Hermes était en arrière-plan.'
    }
  },
  test: 'Envoyer une notification de test',
  testTitle: 'Hermes',
  testBody: 'Les notifications fonctionnent.',
  testSent:
    "Test envoyé. Si rien n'apparaît, vérifiez les autorisations de notification de votre système et le mode Ne pas déranger.",
  testUnsupported: 'Ce système ne prend pas en charge les notifications natives.',
  muteSoundsTitle: 'Couper les sons de l’interface',
  muteSoundsDesc:
    'Coupe les signaux automatiques de fin, d’activation et de réflexion, le retour haptique et les sons des notifications système. Les aperçus sonores et les réponses vocales restent audibles.',
  completionSoundTitle: 'Son de fin',
  completionSoundDesc: "Se joue à la fin d'un tour d'agent. Choisissez un préréglage et prévisualisez-le ici.",
  completionSoundPreview: 'Aperçu'
}
