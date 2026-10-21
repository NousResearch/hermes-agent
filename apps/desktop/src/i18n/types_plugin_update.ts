/** Copy for updating an installed plugin (Plugins tab): catalog re-pin consent and the security-scan override. */
export interface PluginUpdateTranslations {
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
