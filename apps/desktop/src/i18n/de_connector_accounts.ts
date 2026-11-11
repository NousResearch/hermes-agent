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
  renameGone: 'Dieses Konto gibt es nicht mehr. Es wurde entfernt oder durch eine neue Verbindung ersetzt.',
  remove: 'Entfernen',
  removeTitle: (app: string, name: string) => `Das ${app}-Konto ${name} entfernen?`,
  removeBody:
    'Das verbundene Konto wird für alle Agenten entfernt, die dasselbe Nous-Konto verwenden, nicht nur für diesen Agenten. Ihre anderen Konten bleiben verbunden.',
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
