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
  checkForUpdates: 'Buscar actualizaciones',
  checkingForUpdates: 'Comprobando…',
  updatesAvailable: (count: number) =>
    count === 1 ? '1 actualización de plugin disponible' : `${count} actualizaciones de plugins disponibles`,
  checkFailed: 'No se pudieron buscar actualizaciones',
  checkUnsupported: 'Este backend no puede buscar actualizaciones de plugins. Actualiza Hermes para activarlo',
  autoUpdate: 'Actualización automática',
  autoUpdateTip: (hours: number) =>
    `Aplicar las actualizaciones de plugins automáticamente cuando el gateway compruebe (cada ${hours} h). El código nuevo se carga en sesiones nuevas.`,
  autoUpdateFailed: 'No se pudo cambiar la actualización automática de plugins'
}
