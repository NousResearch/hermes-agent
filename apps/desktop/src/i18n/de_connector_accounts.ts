import type { TranslationOverrides } from './define-locale'

export const deConnectorAccounts: NonNullable<TranslationOverrides['connectorsPage']>['accounts'] = {
  heading: 'Konten',
  count: (count: number) => `${count} ${count === 1 ? 'Konto' : 'Konten'}`,
  addAnother: 'Weiteres Konto hinzufügen',
  aliasLabel: 'Kontoname',
  aliasHint: 'Kleinbuchstaben, Ziffern und Bindestriche, höchstens 32. Zum Beispiel arbeit oder privat.',
  aliasInvalid:
    'Verwenden Sie Kleinbuchstaben, Ziffern und Bindestriche, beginnend mit einem Buchstaben oder einer Ziffer.',
  aliasTaken: 'Dieser Name wird bereits verwendet.',
  add: 'Verbinden',
  rename: 'Umbenennen',
  renameLabel: (name: string) => `Neuer Name für ${name}`,
  remove: 'Entfernen',
  removeTitle: (app: string, name: string) => `Das ${app}-Konto ${name} entfernen?`,
  removeBody: 'Hermes handelt nicht mehr über dieses Konto. Ihre anderen Konten bleiben verbunden.',
  retiredCount: (count: number) => `${count} ersetzt`,
  status: {
    expired: 'Zugriff abgelaufen',
    failed: 'Verbindung fehlgeschlagen',
    inactive: 'Inaktiv',
    pending: 'Wartet auf Anmeldung',
    retired: 'Durch erneutes Verbinden ersetzt',
    revoked: 'Zugriff widerrufen'
  }
}
