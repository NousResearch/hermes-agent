import type { Translations } from './types'

// Temporary (incognito) chat copy, composed by en.ts.
export const enTemporaryChat = {
  temporaryChat: {
    title: 'Temporary chat',
    description:
      "Temporary chats don't appear in your history, don't use or update memory, and are deleted when you leave the chat.",
    placeholder: 'Ask anything in a temporary chat',
    use: 'Use temporary chat',
    exit: 'Exit temporary chat',
    badge: 'Temporary',
    badgeHint: "Not saved. It ends when you close it, or after 10 minutes idle while you're away.",
    newChat: 'New temporary chat',
    unsupportedBackend:
      "This Hermes backend doesn't support temporary chats yet. Update it, or turn off temporary mode to chat normally.",
    endChat: 'End temporary chat',
    endTitle: 'End this temporary chat?',
    endDescription: "It's deleted for good — nothing about it is saved.",
    endConfirm: 'End and delete',
    returnTo: 'Back to temporary chat',
    running: 'Still responding…'
  }
} satisfies Pick<Translations, 'temporaryChat'>
