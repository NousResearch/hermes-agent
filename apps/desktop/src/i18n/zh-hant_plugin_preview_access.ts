import type { PluginPreviewAccessTranslations } from './types_plugin_preview_access'

/** Copy for the preview-script consent prompt and the per-plugin grant list (Capabilities ▸ Plugins). */
export const zhHantPluginPreviewAccess: PluginPreviewAccessTranslations = {
  previewScriptPromptTitle: (plugin: string) => `允許 ${plugin} 在此頁面執行指令碼嗎？`,
  previewScriptPromptBody: (plugin: string, origin: string) =>
    `${plugin} 想在預覽瀏覽器中的 ${origin} 上執行 JavaScript。`,
  previewScriptPromptRisk: '指令碼可以讀取並變更該網站上的所有內容，包括你在該網站已登入帳戶的頁面。',
  previewScriptPromptRevokeHint: '僅適用於此網站。你可以在 技能與工具 ▸ 外掛 中撤銷。',
  previewScriptAllow: '在此網站允許',
  previewScriptDeny: '不允許',
  previewScriptAccessLabel: '頁面指令碼',
  previewScriptAccessNone: '在頁面上執行指令碼前會先詢問',
  previewScriptRevoke: (origin: string) => `撤銷對 ${origin} 的存取`,
  previewScriptRevoked: (plugin: string, origin: string) => `${plugin} 不能再於 ${origin} 上執行指令碼。`,
  previewScriptRevokeUnsaved: (plugin: string, origin: string) =>
    `無法儲存此變更，因此它僅在 Hermes 重新啟動前有效。目前 ${plugin} 在 ${origin} 上被封鎖。`
}
