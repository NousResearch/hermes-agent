import type { ProfileScope } from '@/api/client'
import { translateNow } from '@/i18n'
import { lookupPluginCatalogEntry, type PluginCatalogEntry, type PluginCatalogLookupError } from '@/lib/plugin-catalog'

import { $agentPlugins } from './agent-plugins'
import { notify } from './notifications'
import { openPluginInstallRequest } from './plugin-install-request'

/**
 * THE way a curated-catalog pick reaches the Install Plugin dialog. The
 * Plugins tab's embedded picker and the `hermes://plugin/install?catalog=`
 * deep link both land here, so a link opens exactly the reviewed/pinned
 * dialog an in-app pick does: `catalogName` makes the backend resolve the
 * pinned SHA and record provenance; `repo#subdir` is what the dialog inspects.
 */
export function openCatalogPluginInstall(entry: PluginCatalogEntry, profile: ProfileScope): void {
  const existing = $agentPlugins.get().find(row => row.catalog_name === entry.name || row.name === entry.name)

  if (existing && !existing.update_available) {
    notify({ kind: 'success', message: translateNow('skills.plugins.alreadyInstalled', entry.name) })

    return
  }

  openPluginInstallRequest({
    catalogName: entry.name,
    profile,
    repo: entry.subdir ? `${entry.repo}#${entry.subdir}` : entry.repo,
    sha: entry.sha
  })
}

const DEEP_LINK_ERROR_KEYS: Record<PluginCatalogLookupError, string> = {
  invalid_name: 'skills.plugins.deepLinkCatalogInvalidName',
  unavailable: 'skills.plugins.deepLinkCatalogUnavailable',
  unknown: 'skills.plugins.deepLinkCatalogUnknown'
}

/**
 * `hermes://plugin/install?catalog=<name>`: resolve the name against the live
 * catalog and open the dialog in catalog mode for the active profile. Any
 * failure (bad name, catalog unreachable, name not listed) is a clear error
 * toast — the string is never reinterpreted as a git path.
 */
export async function requestPluginCatalogInstallFromDeepLink(
  name: string,
  lookup: typeof lookupPluginCatalogEntry = lookupPluginCatalogEntry
): Promise<void> {
  const result = await lookup(name)

  if (!result.ok) {
    notify({
      kind: 'error',
      title: translateNow('skills.plugins.deepLinkErrorTitle'),
      message: translateNow(DEEP_LINK_ERROR_KEYS[result.error], name.trim())
    })

    return
  }

  openCatalogPluginInstall(result.entry, null)
}
