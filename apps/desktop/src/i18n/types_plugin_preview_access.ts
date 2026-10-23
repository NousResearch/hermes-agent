/** Copy for the `ctx.preview.executeJavaScript` consent: the app-owned prompt a plugin's first
 *  call on a page origin raises, and the per-plugin grant list in Capabilities ▸ Plugins. */
export interface PluginPreviewAccessTranslations {
  previewScriptPromptTitle: (plugin: string) => string
  previewScriptPromptBody: (plugin: string, origin: string) => string
  previewScriptPromptRisk: string
  previewScriptPromptRevokeHint: string
  previewScriptAllow: string
  previewScriptDeny: string
  previewScriptAccessLabel: string
  previewScriptAccessNone: string
  previewScriptRevoke: (origin: string) => string
  previewScriptRevoked: (plugin: string, origin: string) => string
  previewScriptRevokeUnsaved: (plugin: string, origin: string) => string
}
