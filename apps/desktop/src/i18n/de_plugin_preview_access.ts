import type { PluginPreviewAccessTranslations } from './types_plugin_preview_access'

/** Copy for the preview-script consent prompt and the per-plugin grant list (Capabilities ▸ Plugins). */
export const dePluginPreviewAccess: PluginPreviewAccessTranslations = {
  previewScriptPromptTitle: (plugin: string) => `${plugin} Skripte auf dieser Seite ausführen lassen?`,
  previewScriptPromptBody: (plugin: string, origin: string) =>
    `${plugin} möchte im Vorschau-Browser JavaScript auf ${origin} ausführen.`,
  previewScriptPromptRisk:
    'Skripte können alles auf dieser Website lesen und ändern, auch Seiten von Konten, bei denen du dort angemeldet bist.',
  previewScriptPromptRevokeHint:
    'Gilt nur für diese Website. Du kannst den Zugriff unter Fähigkeiten ▸ Plugins widerrufen.',
  previewScriptAllow: 'Auf dieser Website erlauben',
  previewScriptDeny: 'Nicht erlauben',
  previewScriptAccessLabel: 'Seitenskripte',
  previewScriptAccessNone: 'Fragt vor dem Ausführen von Skripten auf einer Seite',
  previewScriptRevoke: (origin: string) => `Zugriff auf ${origin} widerrufen`,
  previewScriptRevoked: (plugin: string, origin: string) =>
    `${plugin} kann keine Skripte mehr auf ${origin} ausführen.`,
  previewScriptRevokeUnsaved: (plugin: string, origin: string) =>
    `Die Änderung konnte nicht gespeichert werden und gilt nur bis zum Neustart von Hermes. ${plugin} ist auf ${origin} vorerst gesperrt.`
}
