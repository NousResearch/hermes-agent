import type { TranslationOverrides } from './define-locale'

export const zhHantMcpReload: NonNullable<TranslationOverrides['connectorsPage']>['reload'] = {
  action: '重新載入 MCP 伺服器',
  title: '重新載入 MCP 伺服器？',
  description: '這會重新啟動此設定檔的 MCP 伺服器，下一則訊息會重新傳送完整提示，消耗提示快取的 token。',
  confirm: '立即重新載入',
  success: 'MCP 伺服器已重新載入',
  failed: servers => `MCP 伺服器已重新載入，但以下伺服器未連線：${servers}`,
  error: 'MCP 伺服器未重新載入',
  confirmRequired: '/reload-mcp 需要確認',
  reloadedAlways: 'MCP 伺服器已重新載入 · 此設定檔中的 /reload-mcp 將不再要求確認',
  notConnected: servers => `未連線：${servers}`,
  completionNow: '立即重新載入（本次略過確認）',
  completionAlways: '立即重新載入，並在此設定檔中不再詢問',
  skillsFailed: '技能重新載入失敗'
}
