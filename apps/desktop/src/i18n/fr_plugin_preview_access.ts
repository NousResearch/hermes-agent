import type { PluginPreviewAccessTranslations } from './types_plugin_preview_access'

/** Copy for the preview-script consent prompt and the per-plugin grant list (Capabilities ▸ Plugins). */
export const frPluginPreviewAccess: PluginPreviewAccessTranslations = {
  previewScriptPromptTitle: (plugin: string) => `Autoriser ${plugin} à exécuter des scripts sur cette page ?`,
  previewScriptPromptBody: (plugin: string, origin: string) =>
    `${plugin} veut exécuter du JavaScript dans le navigateur d’aperçu sur ${origin}.`,
  previewScriptPromptRisk: 'Les scripts peuvent lire et modifier tout ce qui se trouve sur ce site, y compris les pages des comptes auxquels vous y êtes connecté.',
  previewScriptPromptRevokeHint: 'S’applique à ce site uniquement. Vous pouvez révoquer l’accès dans Capacités ▸ Plugins.',
  previewScriptAllow: 'Autoriser sur ce site',
  previewScriptDeny: 'Ne pas autoriser',
  previewScriptAccessLabel: 'Scripts de page',
  previewScriptAccessNone: 'Demande avant d’exécuter des scripts sur une page',
  previewScriptRevoke: (origin: string) => `Révoquer l’accès à ${origin}`,
  previewScriptRevoked: (plugin: string, origin: string) => `${plugin} ne peut plus exécuter de scripts sur ${origin}.`,
  previewScriptRevokeUnsaved: (plugin: string, origin: string) =>
    `Impossible d’enregistrer la modification : elle ne vaut que jusqu’au redémarrage de Hermes. ${plugin} est bloqué sur ${origin} pour l’instant.`
}
