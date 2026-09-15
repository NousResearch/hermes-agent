export type VaultKind = 'address' | 'login' | 'payment'
export const VAULT_KINDS: readonly VaultKind[] = ['login', 'payment', 'address']
export const IDENTIFIER_TYPES = ['email', 'phone', 'username'] as const
export type IdentifierType = (typeof IDENTIFIER_TYPES)[number]

/** Add-dialog prefill from a deep link (`/settings?tab=vault&kind=…`). NEVER secrets. */
export interface VaultPrefill {
  kind?: string
  label?: string
  origin?: string
}

export function isVaultKind(value: string | undefined): value is VaultKind {
  return !!value && (VAULT_KINDS as readonly string[]).includes(value)
}

export function isValidOrigin(value: string): boolean {
  try {
    const url = new URL(value)

    return (url.protocol === 'https:' || url.protocol === 'http:') && !!url.hostname
  } catch {
    return false
  }
}

export const EMPTY_FORM = {
  kind: 'login' as VaultKind,
  label: '',
  origin: '',
  identifierType: 'email' as IdentifierType,
  identifier: '',
  password: '',
  otpSecret: '',
  cardNumber: '',
  cardName: '',
  expMonth: '',
  expYear: '',
  cvc: '',
  postal: '',
  line1: '',
  line2: '',
  city: '',
  state: '',
  country: ''
}

export type VaultForm = typeof EMPTY_FORM

export function buildSecret(form: VaultForm): Record<string, string> {
  if (form.kind === 'login') {
    // identifier_type/identifier are stored as agent-visible metadata by the
    // vault store; only the password stays in the encrypted secret payload.
    return {
      identifier_type: form.identifierType,
      identifier: form.identifier.trim(),
      password: form.password,
      ...(form.otpSecret.trim() ? { otp_secret: form.otpSecret.trim() } : {})
    }
  }

  if (form.kind === 'payment') {
    return {
      card_number: form.cardNumber.replace(/\s+/g, ''),
      cardholder_name: form.cardName.trim(),
      exp_month: form.expMonth.trim(),
      exp_year: form.expYear.trim(),
      cvc: form.cvc,
      billing_postal_code: form.postal.trim()
    }
  }

  const secret: Record<string, string> = {
    address_line1: form.line1.trim(),
    city: form.city.trim(),
    postal_code: form.postal.trim(),
    country: form.country.trim()
  }

  if (form.line2.trim()) {
    secret.address_line2 = form.line2.trim()
  }

  if (form.state.trim()) {
    secret.state = form.state.trim()
  }

  return secret
}
