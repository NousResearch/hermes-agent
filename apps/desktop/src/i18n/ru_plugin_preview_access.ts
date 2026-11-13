import type { PluginPreviewAccessTranslations } from './types_plugin_preview_access'

/** Copy for the preview-script consent prompt and the per-plugin grant list (Capabilities ▸ Plugins). */
export const ruPluginPreviewAccess: PluginPreviewAccessTranslations = {
  previewScriptPromptTitle: (plugin: string) => `Разрешить ${plugin} запускать скрипты на этой странице?`,
  previewScriptPromptBody: (plugin: string, origin: string) =>
    `${plugin} хочет выполнить JavaScript во встроенном браузере на ${origin}.`,
  previewScriptPromptRisk: 'Скрипты могут читать и изменять всё на этом сайте, включая страницы аккаунтов, в которые вы там вошли.',
  previewScriptPromptRevokeHint: 'Действует только для этого сайта. Отозвать можно в разделе Возможности ▸ Плагины.',
  previewScriptAllow: 'Разрешить на этом сайте',
  previewScriptDeny: 'Не разрешать',
  previewScriptAccessLabel: 'Скрипты страниц',
  previewScriptAccessNone: 'Спрашивает перед запуском скриптов на странице',
  previewScriptRevoke: (origin: string) => `Отозвать доступ к ${origin}`,
  previewScriptRevoked: (plugin: string, origin: string) => `${plugin} больше не может запускать скрипты на ${origin}.`,
  previewScriptRevokeUnsaved: (plugin: string, origin: string) =>
    `Не удалось сохранить изменение, поэтому оно действует только до перезапуска Hermes. Пока ${plugin} заблокирован на ${origin}.`
}
