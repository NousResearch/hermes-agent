import type { PluginUpdateAllTranslations } from './types_plugin_update'

/** Aba Plugins: "Atualizar todos" e opção de atualização automática. */
export const ptBrPluginUpdateAll: PluginUpdateAllTranslations = {
  updateAll: 'Atualizar todos',
  updatingAll: 'Atualizando…',
  updateAllTip: 'Verificar todos os plugins instalados e aplicar as atualizações que não exigem revisão',
  updateAllDone: (updated: number, review: number, failed: number) =>
    `Atualizados: ${updated}` + (review ? `, ${review} para revisar` : '') + (failed ? `, ${failed} com falha` : ''),
  updateAllNone: 'Todos os plugins estão atualizados',
  updateAllFailed: 'Não foi possível atualizar os plugins',
  autoUpdate: 'Atualização automática',
  autoUpdateTip: (hours: number) =>
    `Aplicar atualizações de plugins automaticamente quando o gateway verificar (a cada ${hours} h). O código atualizado é carregado em novas sessões.`,
  autoUpdateFailed: 'Não foi possível alterar a atualização automática de plugins'
}
