export interface VaultTranslations {
  title: string
  blurb: string
  count: (n: number) => string
  loadFailed: string
  empty: string
  emptyDesc: string
  add: string
  addTitle: string
  addDescription: string
  added: string
  adding: string
  addConfirm: string
  kindField: string
  kinds: Record<'address' | 'login' | 'payment', string>
  labelField: string
  labelPlaceholder: string
  labelRequired: string
  originField: string
  originPlaceholder: string
  originPlaceholderCheckout: string
  originInvalid: string
  originAnySiteHint: string
  anySite: string
  identifierTypeField: string
  identifierTypes: Record<'email' | 'phone' | 'username', string>
  identifierField: string
  identifierShown: (identifier: string) => string
  passwordField: string
  loginFieldsRequired: string
  cardNumberField: string
  cardNameField: string
  expMonthField: string
  expYearField: string
  cvcField: string
  postalField: string
  addressLine1Field: string
  addressLine2Field: string
  cityField: string
  stateField: string
  countryField: string
  optional: string
  createdOn: (date: string) => string
  deleteAction: string
  otpField: string
  otpPlaceholder: string
  otpHint: string
  twoFactorBadge: string
  deleteTitle: string
  deleteDescription: (label: string) => string
  deleteConfirm: string
  sources: {
    title: string
    blurb: string
    toggleFailed: string
    notInstalled: (name: string) => string
    disabledDesc: string
    lockedDesc: string
    unlockedDesc: string
    statusLocked: string
    statusNotDetected: string
    statusOff: string
    statusUnlocked: string
    unlock: string
    unlocking: string
    lock: string
    unlocked: (name: string) => string
    unlockTitle: (name: string) => string
    unlockDescription: string
    masterPasswordPlaceholder: string
  }
}
