/** Plugins tab copy for the folders a plugin may use (`ctx.files`): the app's
 *  picker, the per-plugin grant list and its Revoke action. */
export interface PluginFolderAccessTranslations {
  title: string
  hint: (name: string) => string
  revoke: string
  revokeLabel: (path: string) => string
  revoked: (name: string, path: string) => string
  revokeFailed: (name: string) => string
  pickerTitle: (name: string) => string
  pickerButton: string
  refusedProtected: (name: string) => string
  loadFailed: (name: string) => string
}

/** Copy for updating an installed plugin (Plugins tab): catalog re-pin consent and the security-scan override,
 *  plus the plugin's folder access. */
export interface PluginUpdateTranslations {
  folderAccess: PluginFolderAccessTranslations
  updateToPin: (sha: string) => string
  updateFailed: (name: string) => string
  updated: (name: string) => string
  updateConsentTitle: (name: string) => string
  updateConsentBody: (name: string, sha: string) => string
  updateConsentConfirm: string
  updateScanTitle: (name: string) => string
  updateScanBody: (name: string, count: number) => string
  updateScanMore: (count: number) => string
  updateScanConfirm: string
  updateScanBlocked: (name: string) => string
  updatedGit: (name: string) => string
}

/** Copy for the Plugins tab's "Update all" action and its auto-update switch (`plugins.auto_apply`). */
export interface PluginUpdateAllTranslations {
  updateAll: string
  updatingAll: string
  updateAllTip: string
  updateAllDone: (updated: number, review: number, failed: number) => string
  updateAllNone: string
  updateAllFailed: string
  autoUpdate: string
  autoUpdateTip: (hours: number) => string
  autoUpdateFailed: string
}
