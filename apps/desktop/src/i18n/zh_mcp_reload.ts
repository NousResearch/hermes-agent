import type { TranslationOverrides } from './define-locale'

export const zhMcpReload: NonNullable<TranslationOverrides['connectorsPage']>['reload'] = {
  action: '重新加载 MCP 服务器',
  title: '重新加载 MCP 服务器？',
  description:
    '这会将此配置文件的 MCP 服务器与配置同步。如果此聊天的工具发生变化，下一条消息会开始新的提示缓存前缀；没有变化的重新加载不会重新发送提示。',
  confirm: '立即重新加载',
  deferred:
    '此配置文件的 MCP 服务器已更新。此聊天会保留当前工具，直到运行中的回合结束；届时请再次重新加载（或重新打开聊天）。',
  unchanged: 'MCP 服务器已是最新',
  success: 'MCP 服务器已重新加载',
  failed: servers => `MCP 服务器已重新加载，但以下服务器未连接：${servers}`,
  error: 'MCP 服务器未重新加载',
  confirmRequired: '/reload-mcp 需要确认',
  reloadedAlways: 'MCP 服务器已重新加载 · 此配置文件中的 /reload-mcp 将不再要求确认',
  notConnected: servers => `未连接：${servers}`,
  completionNow: '立即重新加载（本次跳过确认）',
  completionAlways: '立即重新加载，并在此配置文件中不再询问',
  skillsRemote: '技能列表已刷新。实时重新扫描在拥有此配置文件的网关上进行，而不是在此窗口中。',
  skillsFailed: '技能重新加载失败'
}
