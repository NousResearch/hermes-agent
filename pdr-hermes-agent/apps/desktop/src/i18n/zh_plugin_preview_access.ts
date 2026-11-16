import type { PluginPreviewAccessTranslations } from './types_plugin_preview_access'

/** Copy for the preview-script consent prompt and the per-plugin grant list (Capabilities ▸ Plugins). */
export const zhPluginPreviewAccess: PluginPreviewAccessTranslations = {
  previewScriptPromptTitle: (plugin: string) => `允许 ${plugin} 在此页面运行脚本吗？`,
  previewScriptPromptBody: (plugin: string, origin: string) =>
    `${plugin} 想在预览浏览器中的 ${origin} 上运行 JavaScript。`,
  previewScriptPromptRisk: '脚本可以读取并更改该网站上的所有内容，包括你在该网站已登录账户的页面。',
  previewScriptPromptRevokeHint: '仅适用于此网站。你可以在 技能与工具 ▸ 插件 中撤销。',
  previewScriptAllow: '在此网站允许',
  previewScriptDeny: '不允许',
  previewScriptAccessLabel: '页面脚本',
  previewScriptAccessNone: '在页面上运行脚本前会先询问',
  previewScriptRevoke: (origin: string) => `撤销对 ${origin} 的访问`,
  previewScriptRevoked: (plugin: string, origin: string) => `${plugin} 不能再在 ${origin} 上运行脚本。`,
  previewScriptRevokeUnsaved: (plugin: string, origin: string) =>
    `无法保存此更改，因此它仅在 Hermes 重启前有效。目前 ${plugin} 在 ${origin} 上被阻止。`
}
