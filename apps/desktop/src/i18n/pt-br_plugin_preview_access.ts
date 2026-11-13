import type { PluginPreviewAccessTranslations } from './types_plugin_preview_access'

/** Textos da autorização de scripts na prévia e das permissões por plugin (Recursos ▸ Plugins). */
export const ptBrPluginPreviewAccess: PluginPreviewAccessTranslations = {
  previewScriptPromptTitle: (plugin: string) => `Permitir que ${plugin} execute scripts nesta página?`,
  previewScriptPromptBody: (plugin: string, origin: string) =>
    `${plugin} quer executar JavaScript no navegador de prévia em ${origin}.`,
  previewScriptPromptRisk:
    'Os scripts podem ler e alterar tudo nesse site, incluindo páginas de contas nas quais você está conectado.',
  previewScriptPromptRevokeHint: 'Aplica-se apenas a este site. Você pode revogar em Recursos ▸ Plugins.',
  previewScriptAllow: 'Permitir neste site',
  previewScriptDeny: 'Não permitir',
  previewScriptAccessLabel: 'Scripts de página',
  previewScriptAccessNone: 'Pede autorização antes de executar scripts em uma página',
  previewScriptRevoke: (origin: string) => `Revogar acesso a ${origin}`,
  previewScriptRevoked: (plugin: string, origin: string) => `${plugin} não pode mais executar scripts em ${origin}.`,
  previewScriptRevokeUnsaved: (plugin: string, origin: string) =>
    `Não foi possível salvar essa alteração, então ela só vale até o Hermes reiniciar. ${plugin} está bloqueado em ${origin} por enquanto.`
}
