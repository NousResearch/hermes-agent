import type { Translations } from './types'

export const deVault: Translations['settings']['vault'] = {
  title: 'Passwörter & Logins',
  blurb:
    'Sagen Sie „melde mich bei GitHub an“, und der Agent meldet Sie an. Beim ersten Mal auf einer Anmeldeseite fragt er Sie direkt dort nach dem Login; danach läuft es einfach. Passwörter sind auf diesem Rechner verschlüsselt und werden direkt in die Seite eingetragen – das Modell sieht sie nie.',
  count: n => `${n} gespeichert`,
  loadFailed: 'Gespeicherte Einträge konnten nicht geladen werden',
  empty: 'Noch nichts gespeichert',
  emptyDesc:
    'Sie müssen hier nichts eintragen. Bitten Sie den Agenten, sich bei einer Seite anzumelden – er fragt Sie dann einmal direkt dort nach dem Login. Über „Hinzufügen“ können Sie einen Eintrag auch vorab anlegen.',
  add: 'Hinzufügen',
  addTitle: 'Login, Karte oder Adresse hinzufügen',
  addDescription: 'Verschlüsselt auf diesem Rechner gespeichert. Der Agent sieht das Passwort nie.',
  added: 'Gespeichert.',
  adding: 'Wird gespeichert…',
  addConfirm: 'Speichern',
  kindField: 'Art',
  kinds: {
    login: 'Login',
    payment: 'Zahlungskarte',
    address: 'Adresse'
  },
  labelField: 'Bezeichnung',
  labelPlaceholder: 'z. B. GitHub Firma',
  labelRequired: 'Eine Bezeichnung ist erforderlich.',
  originField: 'Ursprung der Seite',
  originPlaceholder: 'https://github.com',
  originPlaceholderCheckout: 'https://shop.example.com',
  originInvalid: 'Geben Sie eine gültige URL wie https://example.com ein.',
  originAnySiteHint:
    'Leer lassen, um diese Adresse auf jeder Website zu verwenden; der Agent bittet Sie bei jedem Ausfüllen um Bestätigung der Website.',
  anySite: 'Jede Website',
  identifierTypeField: 'Art der Kennung',
  identifierTypes: {
    email: 'E-Mail',
    phone: 'Telefon',
    username: 'Benutzername'
  },
  identifierField: 'Kennung',
  identifierShown: identifier => identifier,
  passwordField: 'Passwort',
  loginFieldsRequired: 'Kennung und Passwort sind erforderlich.',
  cardNumberField: 'Kartennummer',
  cardNameField: 'Name auf der Karte',
  expMonthField: 'Ablaufmonat',
  expYearField: 'Ablaufjahr',
  cvcField: 'CVC',
  postalField: 'Postleitzahl',
  addressLine1Field: 'Adresszeile 1',
  addressLine2Field: 'Adresszeile 2',
  cityField: 'Stadt',
  stateField: 'Bundesland / Region',
  countryField: 'Land',
  optional: '(optional)',
  createdOn: date => `Hinzugefügt am ${date}`,
  deleteAction: 'Gespeicherten Eintrag entfernen',
  otpField: 'Authentifizierungsschlüssel',
  otpPlaceholder: 'Base32-Geheimnis oder otpauth://-Link',
  otpHint:
    'Der „Einrichtungsschlüssel", den die Seite beim Aktivieren von 2FA anzeigt. Ist er gespeichert, erzeugt Hermes die Codes selbst.',
  twoFactorBadge: '2FA automatisch',
  deleteTitle: 'Diesen Eintrag löschen?',
  deleteDescription: label => `„${label}" wird entfernt. Das kann nicht rückgängig gemacht werden.`,
  deleteConfirm: 'Löschen',
  sources: {
    title: 'Passwortmanager',
    blurb:
      'Installierte Passwortmanager werden automatisch erkannt. Der Agent bittet Sie, einen zu entsperren, wenn er zum ersten Mal einen Login daraus braucht (einmal pro Session); nur ein Session-Token bleibt im Speicher, und der Agent sieht weder Ihr Master-Passwort noch einen Login.',
    toggleFailed: 'Passwortmanager konnte nicht geändert werden',
    notInstalled: name =>
      `Nicht erkannt. Installieren Sie das ${name}-Kommandozeilenwerkzeug und melden Sie sich dort an; Hermes erkennt es automatisch.`,
    disabledDesc: 'Erkannt, aber für Hermes ausgeschaltet.',
    lockedDesc:
      'Erkannt. Der Agent bittet Sie, ihn zu entsperren, wenn er einen Login braucht – oder entsperren Sie ihn jetzt.',
    unlockedDesc:
      'Für diese Session entsperrt. Sperrt automatisch nach 30 Minuten Inaktivität oder wenn Hermes geschlossen wird.',
    statusLocked: 'Gesperrt',
    statusNotDetected: 'Nicht erkannt',
    statusOff: 'Aus',
    statusUnlocked: 'Entsperrt',
    unlock: 'Entsperren',
    unlocking: 'Wird entsperrt…',
    lock: 'Sperren',
    unlocked: name => `${name} ist für diese Session entsperrt.`,
    unlockTitle: name => `${name} entsperren`,
    unlockDescription:
      'Geben Sie Ihr Master-Passwort ein. Es geht an den Passwortmanager auf diesem Rechner und wird danach verworfen – es wird nie gespeichert, protokolliert oder dem Agenten gezeigt.',
    masterPasswordPlaceholder: 'Master-Passwort'
  }
}
