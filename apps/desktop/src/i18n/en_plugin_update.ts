import type { PluginUpdateTranslations } from './types_plugin_update'

/** Copy for updating an installed plugin (Plugins tab): catalog re-pin consent and the security-scan override. */
export const enPluginUpdate: PluginUpdateTranslations = {
  updateToPin: (sha: string) => `Update to ${sha}`,
  updateFailed: (name: string) => `Could not update ${name}`,
  updated: (name: string) => `${name} updated to the current catalog pin. Restart the gateway to apply.`,
  updateConsentTitle: (name: string) => `${name} asks for more`,
  updateConsentBody: (name: string, sha: string) =>
    `The new catalog pin of ${name} (${sha}) adds surfaces the installed version does not have. Apply it only if you trust them:`,
  updateConsentConfirm: 'Apply update',
  updateScanTitle: (name: string) => `Security scan flagged ${name}`,
  updateScanBody: (name: string, count: number) =>
    `The update to ${name} got a caution verdict (${count} ${count === 1 ? 'finding' : 'findings'}). Every update is scanned. Apply it only if you trust the source:`,
  updateScanMore: (count: number) => `+ ${count} more`,
  updateScanConfirm: 'Update anyway',
  updateScanBlocked: (name: string) =>
    `The security scan blocked the update to ${name} (dangerous verdict). It cannot be overridden.`,
  updatedGit: (name: string) => `${name} updated to the latest commit.`
}
