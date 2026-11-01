import type { TranslationOverrides } from './define-locale'

export const frConnectorAccounts: NonNullable<TranslationOverrides['connectorsPage']>['accounts'] = {
  heading: 'Comptes',
  count: (count: number) => `${count} compte${count === 1 ? '' : 's'}`,
  addAnother: 'Ajouter un autre compte',
  aliasLabel: 'Nom du compte',
  aliasHint: 'Lettres minuscules, chiffres et tirets, 32 au plus. Par exemple travail ou maison.',
  localHint:
    'Crée une connexion MCP directe distincte, pas un connecteur hébergé. Ce nom identifie ses outils ; choisissez le compte ou l’espace de travail voulu sur la page de connexion du fournisseur. En cas d’échec ou d’annulation, la nouvelle entrée reste disponible pour réessayer ou la retirer.',
  aliasInvalid:
    'Utilisez des lettres minuscules, des chiffres et des tirets, en commençant par une lettre ou un chiffre.',
  aliasTaken: 'Ce nom est déjà utilisé.',
  add: 'Connecter',
  rename: 'Renommer',
  renameLabel: (name: string) => `Nouveau nom pour ${name}`,
  renameGone: 'Ce compte n’existe plus. Il a été supprimé ou remplacé par une reconnexion.',
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
