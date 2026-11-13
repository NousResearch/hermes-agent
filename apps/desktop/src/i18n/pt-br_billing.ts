import type { BillingTranslations } from './types_billing'

export const ptBrBilling: BillingTranslations = {
  billingBlock: {
    titleNous: 'Seus créditos da Nous acabaram',
    titleProvider: provider => `Sem créditos — ${provider}`,
    fallbackMessage: 'Sua conta está sem créditos. Adicione créditos para continuar.',
    openBilling: 'Abrir cobrança',
    addCredits: 'Adicionar créditos',
    dismiss: 'Dispensar'
  }
}
