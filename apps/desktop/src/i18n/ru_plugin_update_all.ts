import type { PluginUpdateAllTranslations } from './types_plugin_update'

/** Plugins tab: "Update all" and the auto-update switch. */
export const ruPluginUpdateAll: PluginUpdateAllTranslations = {
  updateAll: 'Обновить все',
  updatingAll: 'Обновление…',
  updateAllTip: 'Проверить все установленные плагины и применить обновления, не требующие проверки',
  updateAllDone: (updated: number, review: number, failed: number) =>
    `Обновлено: ${updated}` +
    (review ? `, требуют проверки: ${review}` : '') +
    (failed ? `, с ошибкой: ${failed}` : ''),
  updateAllNone: 'Все плагины в актуальном состоянии',
  updateAllFailed: 'Не удалось обновить плагины',
  checkForUpdates: 'Проверить обновления',
  checkingForUpdates: 'Проверка…',
  updatesAvailable: (count: number) => `Доступно обновлений плагинов: ${count}`,
  checkFailed: 'Не удалось проверить обновления',
  checkUnsupported: 'Этот бэкенд не умеет проверять обновления плагинов. Обновите Hermes, чтобы включить это',
  autoUpdate: 'Автообновление',
  autoUpdateTip: (hours: number) =>
    `Автоматически применять обновления плагинов при проверке шлюзом (каждые ${hours} ч). Новый код загружается в новых сессиях.`,
  autoUpdateFailed: 'Не удалось изменить автообновление плагинов'
}
