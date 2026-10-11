import type { TranslationOverrides } from './define-locale'

export const zhMcpReload: NonNullable<TranslationOverrides['connectorsPage']>['reload'] = {
  action: '重新加载 MCP 服务器',
  title: '重新加载 MCP 服务器？',
  description: '这将重启此配置文件的 MCP 服务器，下一条消息会重新发送完整提示，消耗提示缓存的 token。',
  confirm: '立即重新加载',
  success: 'MCP 服务器已重新加载',
  failed: servers => `MCP 服务器已重新加载，但以下服务器未连接：${servers}`,
  error: 'MCP 服务器未重新加载',
  confirmRequired: '/reload-mcp 需要确认',
  reloadedAlways: 'MCP 服务器已重新加载 · 此配置文件中的 /reload-mcp 将不再要求确认',
  notConnected: servers => `未连接：${servers}`,
  completionNow: '立即重新加载（本次跳过确认）',
  completionAlways: '立即重新加载，并在此配置文件中不再询问',
  skillsFailed: '技能重新加载失败'
}
