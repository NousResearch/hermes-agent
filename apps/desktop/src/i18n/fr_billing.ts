import type { BillingTranslations } from './types_billing'

// Out-of-credits overlay copy (Nous vs. third-party providers), spread into fr.ts.
export const frBilling: BillingTranslations = {
  billingBlock: {
    titleNous: 'Crédits Nous épuisés',
    titleProvider: provider => `Crédits épuisés — ${provider}`,
    fallbackMessage: 'Votre compte est épuisé. Ajoutez des crédits pour continuer.',
    openBilling: 'Ouvrir la facturation',
    addCredits: 'Ajouter des crédits',
    dismiss: 'Ignorer'
  }
}