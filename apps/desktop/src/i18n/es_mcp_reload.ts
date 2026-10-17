import type { TranslationOverrides } from './define-locale'
import { esConnectorAccounts } from './es_connector_accounts'

export const esMcpReload: NonNullable<TranslationOverrides['connectorsPage']>['reload'] = {
  action: 'Recargar servidores MCP',
  title: '¿Recargar servidores MCP?',
  description:
    'Esto reconcilia los servidores MCP de este perfil con la configuración. Si cambian las herramientas de este chat, el próximo mensaje inicia un nuevo prefijo de caché de prompt; una recarga sin cambios no reenvía el prompt.',
  confirm: 'Recargar ahora',
  deferred:
    'Servidores MCP actualizados para este perfil. Este chat conserva sus herramientas hasta que termine su turno en curso; vuelve a recargar entonces (o reabre el chat).',
  unchanged: 'Los servidores MCP ya están actualizados',
  success: 'Servidores MCP recargados',
  failed: servers => `Servidores MCP recargados, pero estos no se conectaron: ${servers}`,
  error: 'Los servidores MCP no se recargaron',
  confirmRequired: '/reload-mcp requiere confirmación',
  reloadedAlways: 'Servidores MCP recargados · /reload-mcp ya no pedirá confirmación en este perfil',
  notConnected: servers => `No se conectaron: ${servers}`,
  completionNow: 'Recargar ahora (omitir la confirmación una vez)',
  completionAlways: 'Recargar ahora y no volver a preguntar en este perfil',
  skillsRemote:
    'Lista de skills actualizada. El reanálisis en vivo se ejecuta en la puerta de enlace propietaria de este perfil, no desde esta ventana.',
  skillsFailed: 'No se pudieron recargar los skills'
}

/** The connectors-page slice this file owns: the accounts list plus the reload copy, so the
 *  locale facade stays one spread (its line budget only goes down). */
export const esConnectorPage: NonNullable<TranslationOverrides['connectorsPage']> = {
  accounts: esConnectorAccounts,
  reload: esMcpReload
}
