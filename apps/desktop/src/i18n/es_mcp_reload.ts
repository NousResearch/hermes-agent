import type { TranslationOverrides } from './define-locale'

export const esMcpReload: NonNullable<TranslationOverrides['connectorsPage']>['reload'] = {
  action: 'Recargar servidores MCP',
  title: '¿Recargar servidores MCP?',
  description:
    'Esto reinicia los servidores MCP de este perfil y el próximo mensaje reenvía el prompt completo, lo que consume tokens de la caché de prompt.',
  confirm: 'Recargar ahora',
  success: 'Servidores MCP recargados',
  failed: servers => `Servidores MCP recargados, pero estos no se conectaron: ${servers}`,
  error: 'Los servidores MCP no se recargaron',
  confirmRequired: '/reload-mcp requiere confirmación',
  reloadedAlways: 'Servidores MCP recargados · /reload-mcp ya no pedirá confirmación en este perfil',
  notConnected: servers => `No se conectaron: ${servers}`,
  completionNow: 'Recargar ahora (omitir la confirmación una vez)',
  completionAlways: 'Recargar ahora y no volver a preguntar en este perfil',
  skillsFailed: 'No se pudieron recargar los skills'
}
