import type { PluginUpdateAllTranslations } from './types_plugin_update'

/** Plugins tab: "Update all" and the auto-update switch. */
export const enPluginUpdateAll: PluginUpdateAllTranslations = {
  updateAll: 'Update all',
  updatingAll: 'Updating…',
  updateAllTip: 'Check every installed plugin and apply the updates that need no review',
  updateAllDone: (updated: number, review: number, failed: number) =>
    `Updated ${updated}` + (review ? `, ${review} need review` : '') + (failed ? `, ${failed} failed` : ''),
  updateAllNone: 'All plugins are up to date',
  updateAllFailed: 'Could not update plugins',
  autoUpdate: 'Auto-update',
  autoUpdateTip: (hours: number) =>
    `Apply plugin updates automatically when Hermes checks (every ${hours} h). Updated code loads in new sessions.`,
  autoUpdateFailed: 'Could not change plugin auto-update'
}
