import type { TranslationOverrides } from './define-locale'

export const ruMcpReload: NonNullable<TranslationOverrides['connectorsPage']>['reload'] = {
  action: 'Перезагрузить MCP-серверы',
  title: 'Перезагрузить MCP-серверы?',
  description:
    'Это перезапустит MCP-серверы этого профиля, а следующее сообщение заново отправит полный промпт, что расходует токены кэша промптов.',
  confirm: 'Перезагрузить сейчас',
  success: 'MCP-серверы перезагружены',
  failed: servers => `MCP-серверы перезагружены, но эти не подключились: ${servers}`,
  error: 'MCP-серверы не перезагружены',
  confirmRequired: '/reload-mcp требует подтверждения',
  reloadedAlways: 'MCP-серверы перезагружены · в этом профиле /reload-mcp больше не будет запрашивать подтверждение',
  notConnected: servers => `Не подключились: ${servers}`,
  completionNow: 'Перезагрузить сейчас (один раз без подтверждения)',
  completionAlways: 'Перезагрузить сейчас и больше не спрашивать в этом профиле',
  skillsFailed: 'Не удалось перезагрузить навыки'
}
