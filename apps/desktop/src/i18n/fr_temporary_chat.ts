import type { TranslationOverrides } from './define-locale'

// Temporary (incognito) chat copy, composed by fr.ts.
export const frTemporaryChat = {
  temporaryChat: {
    title: 'Discussion temporaire',
    description:
      "Les discussions temporaires n'apparaissent pas dans votre historique, n'utilisent ni ne modifient la mémoire et sont supprimées lorsque vous les quittez.",
    placeholder: 'Posez votre question dans une discussion temporaire',
    use: 'Utiliser une discussion temporaire',
    exit: 'Quitter la discussion temporaire',
    badge: 'Temporaire',
    badgeHint:
      "Non enregistrée. Elle se termine quand vous la fermez, ou après 10 minutes d'inactivité pendant votre absence.",
    newChat: 'Nouvelle discussion temporaire',
    unsupportedBackend:
      'Ce backend Hermes ne prend pas encore en charge les discussions temporaires. Mettez-le à jour ou désactivez le mode temporaire pour discuter normalement.',
    endChat: 'Terminer la discussion temporaire',
    endTitle: 'Terminer cette discussion temporaire ?',
    endDescription: "Elle est supprimée définitivement — rien n'en est conservé.",
    endConfirm: 'Terminer et supprimer',
    returnTo: 'Retour à la discussion temporaire',
    running: 'Réponse en cours…'
  }
} satisfies TranslationOverrides
