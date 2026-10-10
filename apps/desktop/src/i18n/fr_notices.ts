import type { TranslationOverrides } from './define-locale'

// Shell notices (remote-display toast, butterbar), spread into fr.ts.
export const frNotices = {
  remoteDisplayBanner: {
    message: reason =>
      `Rendu logiciel actif — affichage distant détecté (${reason}). L'accélération GPU est désactivée pour éviter les scintillements.`
  },
  previewDraft: {
    discardTitle: 'Abandonner les modifications non enregistrées ?',
    discardBody: label => `${label} contient des modifications non enregistrées. Fermer l'onglet les supprime.`,
    discardConfirm: 'Abandonner les modifications'
  },
  butterbar: {
    goTo: (index, total) => `Afficher l'avis ${index} sur ${total}`,
    legal: {
      before: "L'utilisation de Hermes Agent est soumise à nos ",
      terms: "Conditions d'utilisation",
      between: ' et à notre ',
      privacy: 'Politique de confidentialité',
      after: '.'
    }
  }
} satisfies Pick<TranslationOverrides, 'remoteDisplayBanner' | 'previewDraft' | 'butterbar'>
