import type { BillingTranslations } from './types_billing'

// Texte de superposition hors-crédits (Nous vs. fournisseurs tiers), intégré dans fr.ts.
export const frBilling: BillingTranslations = {
  billingBlock: {
    titleNous: 'Plus de crédits Nous',
    titleProvider: provider => `Plus de crédits — ${provider}`,
    fallbackMessage: 'Votre compte est à court de crédits. Ajoutez-en pour continuer.',
    openBilling: 'Ouvrir la facturation',
    addCredits: 'Ajouter des crédits',
    dismiss: 'Fermer'
  }
}
