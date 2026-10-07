import type { TranslationOverrides } from './define-locale'

// Shell notices (remote-display toast, butterbar), spread into de.ts.
export const deNotices = {
  remoteDisplayBanner: {
    message: reason =>
      `Software-Rendering aktiv — Remote-Display erkannt (${reason}). GPU-Beschleunigung ist deaktiviert, um Flackern zu verhindern.`
  },
  butterbar: {
    goTo: (index, total) => `Hinweis ${index} von ${total} anzeigen`
  }
} satisfies Pick<TranslationOverrides, 'remoteDisplayBanner' | 'butterbar'>
