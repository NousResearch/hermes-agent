import type { PluginUpdateTranslations } from './types_plugin_update'

/** Copy for updating an installed plugin (Plugins tab): catalog re-pin consent and the security-scan override. */
export const enPluginUpdate: PluginUpdateTranslations = {
  folderAccess: {
    title: 'Folder access',
    hint: (name: string) => `${name} can read and change files in these folders. Revoking takes effect at once.`,
    revoke: 'Revoke',
    revokeLabel: (path: string) => `Revoke access to ${path}`,
    revoked: (name: string, path: string) => `${name} can no longer use ${path}.`,
    revokeFailed: (name: string) => `Could not revoke folder access for ${name}`,
    pickerTitle: (name: string) => `Choose a folder for ${name}`,
    pickerButton: 'Allow access',
    refusedProtected: (name: string) =>
      `${name} cannot use that folder: it holds Hermes settings, keys or plugin data. Choose a folder that does not contain them.`,
    loadFailed: (name: string) => `Could not load the folders shared with ${name}.`
  },
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
  updatedGit: (name: string) => `${name} updated to the latest commit.`,
}
