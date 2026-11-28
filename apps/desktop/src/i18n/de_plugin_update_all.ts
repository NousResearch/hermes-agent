import type { PluginUpdateAllTranslations } from './types_plugin_update'

/** Plugins tab: "Update all" and the auto-update switch. */
export const dePluginUpdateAll: PluginUpdateAllTranslations = {
  updateAll: 'Alle aktualisieren',
  updatingAll: 'Wird aktualisiert…',
  updateAllTip: 'Alle installierten Plugins prüfen und Updates anwenden, die keine Prüfung brauchen',
  updateAllDone: (updated: number, review: number, failed: number) =>
    `${updated} aktualisiert` + (review ? `, ${review} zu prüfen` : '') + (failed ? `, ${failed} fehlgeschlagen` : ''),
  updateAllNone: 'Alle Plugins sind aktuell',
  updateAllFailed: 'Plugins konnten nicht aktualisiert werden',
  checkForUpdates: 'Nach Updates suchen',
  checkingForUpdates: 'Wird geprüft…',
  updatesAvailable: (count: number) => `${count} Plugin-Update${count === 1 ? '' : 's'} verfügbar`,
  checkFailed: 'Updates konnten nicht geprüft werden',
  checkUnsupported: 'Dieses Backend kann nicht nach Plugin-Updates suchen. Hermes aktualisieren, um dies zu aktivieren',
  autoUpdate: 'Automatisch aktualisieren',
  autoUpdateTip: (hours: number) =>
    `Plugin-Updates automatisch anwenden, wenn das Gateway prüft (alle ${hours} h). Neuer Code wird in neuen Sitzungen geladen.`,
  autoUpdateFailed: 'Automatische Plugin-Updates konnten nicht geändert werden'
}
