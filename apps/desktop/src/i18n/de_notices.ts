import type { TranslationOverrides } from './define-locale'

// Shell notices (remote-display toast, butterbar), spread into de.ts.
export const deNotices = {
  remoteDisplayBanner: {
    message: reason =>
      `Software-Rendering aktiv — Remote-Display erkannt (${reason}). GPU-Beschleunigung ist deaktiviert, um Flackern zu verhindern.`
  },
  butterbar: {
    goTo: (index, total) => `Hinweis ${index} von ${total} anzeigen`,
    legal: {
      before: 'Die Nutzung von Hermes Agent unterliegt unseren ',
      terms: 'Nutzungsbedingungen',
      between: ' und unserer ',
      privacy: 'Datenschutzerklärung',
      after: '.'
    }
  },
  promptNotices: {
    legacySendUnconfirmed:
      'Dieser Server konnte das frühere Senden dieser Nachricht nicht bestätigen; sie wurde möglicherweise bereits ausgeführt. Prüfe die Unterhaltung, bevor du sie erneut sendest.'
  }
} satisfies Pick<TranslationOverrides, 'remoteDisplayBanner' | 'butterbar' | 'promptNotices'>
