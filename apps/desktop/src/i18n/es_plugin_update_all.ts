import type { PluginUpdateAllTranslations } from './types_plugin_update'

/** Plugins tab: "Update all" and the auto-update switch. */
export const esPluginUpdateAll: PluginUpdateAllTranslations = {
  updateAll: 'Actualizar todo',
  updatingAll: 'Actualizando…',
  updateAllTip: 'Comprobar todos los plugins instalados y aplicar las actualizaciones que no requieren revisión',
  updateAllDone: (updated: number, review: number, failed: number) =>
    `${updated} actualizados` + (review ? `, ${review} por revisar` : '') + (failed ? `, ${failed} fallidos` : ''),
  updateAllNone: 'Todos los plugins están al día',
  updateAllFailed: 'No se pudieron actualizar los plugins',
  autoUpdate: 'Actualización automática',
  autoUpdateTip: (hours: number) =>
    `Aplicar las actualizaciones de plugins automáticamente cuando el gateway compruebe (cada ${hours} h). El código nuevo se carga en sesiones nuevas.`,
  autoUpdateFailed: 'No se pudo cambiar la actualización automática de plugins'
}
