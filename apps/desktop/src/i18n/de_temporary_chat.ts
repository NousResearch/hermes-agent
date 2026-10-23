import type { TranslationOverrides } from './define-locale'

// Temporary (incognito) chat copy, composed by de.ts.
export const deTemporaryChat = {
  temporaryChat: {
    title: 'Temporärer Chat',
    description:
      'Temporäre Chats erscheinen nicht in deinem Verlauf, nutzen oder ändern kein Gedächtnis und werden gelöscht, sobald du den Chat verlässt.',
    placeholder: 'Frag etwas im temporären Chat',
    use: 'Temporären Chat verwenden',
    exit: 'Temporären Chat beenden',
    badge: 'Temporär',
    badgeHint:
      'Nicht gespeichert. Endet, wenn du ihn schließt, oder nach 10 Minuten Leerlauf, während du woanders bist.',
    newChat: 'Neuer temporärer Chat',
    unsupportedBackend:
      'Dieses Hermes-Backend unterstützt noch keine temporären Chats. Aktualisiere es oder schalte den temporären Modus aus, um normal zu chatten.',
    endChat: 'Temporären Chat beenden',
    endTitle: 'Diesen temporären Chat beenden?',
    endDescription: 'Er wird endgültig gelöscht – nichts davon wird gespeichert.',
    endConfirm: 'Beenden und löschen',
    returnTo: 'Zurück zum temporären Chat',
    running: 'Antwortet noch…'
  }
} satisfies TranslationOverrides
