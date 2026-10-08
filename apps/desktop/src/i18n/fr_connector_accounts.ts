import type { TranslationOverrides } from './define-locale'

export const frConnectorAccounts: NonNullable<TranslationOverrides['connectorsPage']>['accounts'] = {
  heading: 'Comptes',
  count: (count: number) => `${count} compte${count === 1 ? '' : 's'}`,
  addAnother: 'Ajouter un autre compte',
  aliasLabel: 'Nom du compte',
  aliasHint: 'Lettres minuscules, chiffres et tirets, 32 au plus. Par exemple travail ou maison.',
  aliasInvalid:
    'Utilisez des lettres minuscules, des chiffres et des tirets, en commençant par une lettre ou un chiffre.',
  aliasTaken: 'Ce nom est déjà utilisé.',
  add: 'Connecter',
  rename: 'Renommer',
  renameLabel: (name: string) => `Nouveau nom pour ${name}`,
  remove: 'Retirer',
  removeTitle: (app: string, name: string) => `Retirer le compte ${app} ${name} ?`,
  removeBody: "Hermes cesse d'agir avec ce compte. Vos autres comptes restent connectés.",
  retiredCount: (count: number) => `${count} retiré${count === 1 ? '' : 's'}`,
  status: {
    expired: 'Accès expiré',
    failed: 'Connexion impossible',
    inactive: 'Inactif',
    pending: 'En attente de connexion',
    retired: 'Remplacé par une reconnexion',
    revoked: 'Accès révoqué'
  }
}
