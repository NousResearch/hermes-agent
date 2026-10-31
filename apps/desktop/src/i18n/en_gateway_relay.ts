import type { GatewayRelayTranslations } from './types_gateway_relay'

// Settings → Gateways → Other gateways, spread into en.ts.
export const enGatewayRelay: GatewayRelayTranslations = {
  title: 'Other gateways',
  intro:
    'Open another Hermes gateway from this Webapp. Its profiles and chats join the gateway switcher; requests go through this gateway, signed with its own key, and act as the account you approve there.',
  empty: 'No other gateways yet.',
  addTitle: 'Connect a gateway',
  urlLabel: 'Webapp address',
  labelLabel: 'Name',
  connect: 'Connect',
  connecting: 'Opening the other gateway…',
  connectFailed: 'Could not start connecting that gateway',
  connected: label => `Connected ${label}`,
  denied: 'The other gateway was not connected: the request was denied there.',
  failed: detail => `The other gateway was not connected: ${detail}`,
  remove: 'Remove',
  removeConfirm: label => `Remove ${label}? This gateway stops reaching it, and its grant there is revoked.`,
  removeFailed: 'Could not remove that gateway',
  test: 'Test',
  reachable: 'Reachable',
  unreachable: error => `Not reachable: ${error}`,
  scopes: scopes => `Also allowed: ${scopes}`,
  coreOnly: 'Chats, profiles, sessions and settings only',
  grantsTitle: 'Gateways that reach this one',
  grantsIntro:
    'Gateways you allowed to relay your Webapp here. Anyone signed in to one of them as you can use this machine through it.',
  grantsEmpty: 'No other gateway reaches this one.',
  grantFrom: (origin, when) => `${origin} · allowed ${when}`,
  revoke: 'Revoke',
  revokeConfirm: name => `Revoke ${name}? It can no longer reach this gateway, and its open connections close.`,
  revokeFailed: 'Could not revoke that gateway',
  loadFailed: 'Could not load the gateway relay'
}
