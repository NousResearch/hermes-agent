import { $pluginRecords, installedDesktopCatalogEntries } from '@/contrib/plugins-store'
import { translateNow } from '@/i18n'
import type { PluginCatalogEntry } from '@/lib/plugin-catalog'

import { $agentPlugins } from './agent-plugins'
import { notify } from './notifications'
import { openPluginInstallRequest, type PluginInstallRequest } from './plugin-install-request'

/**
 * THE way a curated-catalog pick reaches the Install Plugin dialog, so every
 * in-app door opens the same reviewed/pinned dialog: `catalogName` makes the
 * backend resolve the pinned SHA and record provenance; `repo#subdir` is what
 * the dialog inspects.
 */
export function openCatalogPluginInstall(
  entry: PluginCatalogEntry,
  profile: null | string,
  options: Pick<PluginInstallRequest, 'stayAfterInstall' | 'tier' | 'title'> = {}
): void {
  const existing = $agentPlugins.get().find(row => row.catalog_name === entry.name || row.name === entry.name)
  // A desktop-only entry has no agent row; its plugin on disk names the entry and commit it came from. At an
  // older commit than the catalog's it is an update: the dialog installs it (Force replaces the old copy).
  const desktop = installedDesktopCatalogEntries($pluginRecords.get())
  const installedAt = existing ? undefined : desktop.get(entry.name)
  const desktopCurrent = Boolean(installedAt && (!entry.sha || installedAt.includes(entry.sha.toLowerCase())))

  if ((existing && !existing.update_available) || desktopCurrent) {
    notify({ kind: 'success', message: translateNow('skills.plugins.alreadyInstalled', entry.name) })

    return
  }

  openPluginInstallRequest({
    catalogName: entry.name,
    profile,
    repo: entry.subdir ? `${entry.repo}#${entry.subdir}` : entry.repo,
    sha: entry.sha,
    summary: entry.summary,
    tier: entry.tier,
    ...options
  })
}
