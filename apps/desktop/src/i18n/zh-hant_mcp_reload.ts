import type { TranslationOverrides } from './define-locale'

export const zhHantMcpReload: NonNullable<TranslationOverrides['connectorsPage']>['reload'] = {
  action: '重新載入 MCP 伺服器',
  title: '重新載入 MCP 伺服器？',
  description:
    '這會將此設定檔的 MCP 伺服器與設定同步，並重新整理此聊天的工具。下一則訊息會重新傳送完整提示，消耗提示快取的 token。',
  confirm: '立即重新載入',
  deferred:
    '此設定檔的 MCP 伺服器已更新。此聊天會保留目前工具，直到執行中的回合結束；屆時請再次重新載入（或重新開啟聊天）。',
  unchanged: 'MCP 伺服器已是最新',
  success: 'MCP 伺服器已重新載入',
  failed: servers => `MCP 伺服器已重新載入，但以下伺服器未連線：${servers}`,
  error: 'MCP 伺服器未重新載入',
  confirmRequired: '/reload-mcp 需要確認',
  reloadedAlways: 'MCP 伺服器已重新載入 · 此設定檔中的 /reload-mcp 將不再要求確認',
  notConnected: servers => `未連線：${servers}`,
  completionNow: '立即重新載入（本次略過確認）',
  completionAlways: '立即重新載入，並在此設定檔中不再詢問',
  skillsRemote: '技能清單已重新整理。即時重新掃描在擁有此設定檔的閘道上執行，而非從此視窗。',
  skillsFailed: '技能重新載入失敗'
}
