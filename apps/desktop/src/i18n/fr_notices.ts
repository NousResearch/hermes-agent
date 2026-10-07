import type { TranslationOverrides } from './define-locale'

// Shell notices (remote-display toast, butterbar), spread into fr.ts.
export const frNotices = {
  remoteDisplayBanner: {
    message: reason =>
      `Rendu logiciel actif — affichage distant détecté (${reason}). L'accélération GPU est désactivée pour éviter les scintillements.`
  },
  butterbar: {
    goTo: (index, total) => `Afficher l'avis ${index} sur ${total}`
  }
} satisfies Pick<TranslationOverrides, 'remoteDisplayBanner' | 'butterbar'>
