import type { GatewayRelayTranslations } from './types_gateway_relay'

// Settings → Gateways → Other gateways: `settings.gateway.relay` in fr.ts.
export const frGatewayRelay: GatewayRelayTranslations = {
  title: 'Autres passerelles',
  intro:
    'Ouvrez une autre passerelle Hermes depuis cette Webapp. Ses profils et ses discussions rejoignent le sélecteur de passerelles ; les requêtes passent par cette passerelle, signées avec sa propre clé, et agissent au nom du compte que vous approuvez là-bas.',
  empty: 'Aucune autre passerelle pour l’instant.',
  addTitle: 'Connecter une passerelle',
  urlLabel: 'Adresse de la Webapp',
  labelLabel: 'Nom',
  connect: 'Connecter',
  connecting: 'Ouverture de l’autre passerelle…',
  connectFailed: 'Impossible de commencer la connexion à cette passerelle',
  connected: label => `${label} connectée`,
  denied: 'L’autre passerelle n’a pas été connectée : la demande y a été refusée.',
  failed: detail => `L’autre passerelle n’a pas été connectée : ${detail}`,
  remove: 'Retirer',
  removeConfirm: label =>
    `Retirer ${label} ? Cette passerelle ne pourra plus l’atteindre, et son autorisation là-bas sera révoquée.`,
  removeFailed: 'Impossible de retirer cette passerelle',
  test: 'Tester',
  reachable: 'Joignable',
  unreachable: error => `Injoignable : ${error}`,
  scopes: scopes => `Également autorisé : ${scopes}`,
  coreOnly: 'Discussions, profils, sessions et réglages uniquement',
  grantsTitle: 'Passerelles qui atteignent celle-ci',
  grantsIntro:
    'Les passerelles que vous avez autorisées à relayer votre Webapp ici. Toute personne connectée à l’une d’elles avec votre compte peut utiliser cette machine par son intermédiaire.',
  grantsEmpty: 'Aucune autre passerelle n’atteint celle-ci.',
  grantFrom: (origin, when) => `${origin} · autorisée ${when}`,
  revoke: 'Révoquer',
  revokeConfirm: name =>
    `Révoquer ${name} ? Elle ne pourra plus atteindre cette passerelle, et ses connexions ouvertes seront fermées.`,
  revokeFailed: 'Impossible de révoquer cette passerelle',
  loadFailed: 'Impossible de charger le relais des passerelles'
}
