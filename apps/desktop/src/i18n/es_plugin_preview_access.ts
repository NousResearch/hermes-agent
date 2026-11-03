import type { PluginPreviewAccessTranslations } from './types_plugin_preview_access'

/** Copy for the preview-script consent prompt and the per-plugin grant list (Capabilities ▸ Plugins). */
export const esPluginPreviewAccess: PluginPreviewAccessTranslations = {
  previewScriptPromptTitle: (plugin: string) => `¿Permitir que ${plugin} ejecute scripts en esta página?`,
  previewScriptPromptBody: (plugin: string, origin: string) =>
    `${plugin} quiere ejecutar JavaScript en el navegador de vista previa en ${origin}.`,
  previewScriptPromptRisk: 'Los scripts pueden leer y cambiar todo en ese sitio, incluidas las páginas de cuentas en las que tienes la sesión iniciada.',
  previewScriptPromptRevokeHint: 'Solo se aplica a este sitio. Puedes revocarlo en Capacidades ▸ Plugins.',
  previewScriptAllow: 'Permitir en este sitio',
  previewScriptDeny: 'No permitir',
  previewScriptAccessLabel: 'Scripts de página',
  previewScriptAccessNone: 'Pregunta antes de ejecutar scripts en una página',
  previewScriptRevoke: (origin: string) => `Revocar el acceso a ${origin}`,
  previewScriptRevoked: (plugin: string, origin: string) => `${plugin} ya no puede ejecutar scripts en ${origin}.`,
  previewScriptRevokeUnsaved: (plugin: string, origin: string) =>
    `No se pudo guardar el cambio, así que solo dura hasta que Hermes se reinicie. ${plugin} queda bloqueado en ${origin} por ahora.`
}
