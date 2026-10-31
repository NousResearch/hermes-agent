import type { GatewayRelayTranslations } from './types_gateway_relay'

// Settings → Gateways → Other gateways: `settings.gateway.relay` in es.ts.
export const esGatewayRelay: GatewayRelayTranslations = {
  title: 'Otros gateways',
  intro:
    'Abre otro gateway de Hermes desde esta Webapp. Sus perfiles y chats se suman al selector de gateways; las solicitudes pasan por este gateway, firmadas con su propia clave, y actúan como la cuenta que apruebes allí.',
  empty: 'Aún no hay otros gateways.',
  addTitle: 'Conectar un gateway',
  urlLabel: 'Dirección de la Webapp',
  labelLabel: 'Nombre',
  connect: 'Conectar',
  connecting: 'Abriendo el otro gateway…',
  connectFailed: 'No se pudo empezar a conectar ese gateway',
  connected: label => `${label} conectado`,
  denied: 'El otro gateway no se conectó: la solicitud se rechazó allí.',
  failed: detail => `El otro gateway no se conectó: ${detail}`,
  remove: 'Quitar',
  removeConfirm: label => `¿Quitar ${label}? Este gateway dejará de alcanzarlo y se revocará su permiso allí.`,
  removeFailed: 'No se pudo quitar ese gateway',
  removeHere: 'Quitar solo aquí',
  removeUnconfirmed: label =>
    `¿Quitar ${label} solo aquí? Su permiso allí sigue vigente hasta que lo revoques en ${label}.`,
  test: 'Probar',
  reachable: 'Accesible',
  unreachable: error => `No accesible: ${error}`,
  scopes: scopes => `También permitido: ${scopes}`,
  coreOnly: 'Solo chats, perfiles, sesiones y ajustes',
  grantsTitle: 'Gateways que llegan a este',
  grantsIntro:
    'Gateways a los que permitiste retransmitir tu Webapp hasta aquí. Quien haya iniciado sesión en uno de ellos con tu cuenta puede usar este equipo a través de él.',
  grantsEmpty: 'Ningún otro gateway llega a este.',
  grantFrom: (origin, when) => `${origin} · permitido ${when}`,
  revoke: 'Revocar',
  revokeConfirm: name => `¿Revocar ${name}? Ya no podrá llegar a este gateway y se cerrarán sus conexiones abiertas.`,
  revokeFailed: 'No se pudo revocar ese gateway',
  loadFailed: 'No se pudo cargar la retransmisión de gateways'
}
