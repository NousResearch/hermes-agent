import { $apiRequestScope, type ProfileScope, type ResolvedOwner } from '@/api/client'
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
export function openCatalogPluginInstall(
  entry: PluginCatalogEntry,
  profile: null | string,
  owner: ResolvedOwner = { ...$apiRequestScope.get(), profile: profile ?? $apiRequestScope.get().profile }
): void {
  const memoryProvider = entry.category === 'memory' && entry.featured === true
  const existing = $agentPlugins.get().find(row => row.catalog_name === entry.name || row.name === entry.name)

  // The Plugins tab cache is not owner-keyed and cannot prove memory installation.
  if (!memoryProvider && existing && !existing.update_available) {
    notify({ kind: 'success', message: translateNow('skills.plugins.alreadyInstalled', entry.name) })

    return
  }

  openPluginInstallRequest({
    catalogName: entry.name,
    profile,
    repo: entry.subdir ? `${entry.repo}#${entry.subdir}` : entry.repo,
    sha: entry.sha,
    ...(memoryProvider
      ? {
          profile: owner.profile,
          legacyHint: 'agent' as const,
          enable: true,
          memory: { name: entry.name, owner }
        }
      : {})
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
  lookup: typeof lookupPluginCatalogEntry = lookupPluginCatalogEntry,
  scope?: ProfileScope
): Promise<void> {
  const ambient = $apiRequestScope.get()
  const profile = typeof scope === 'string' ? scope : (scope?.profile ?? null)

  const owner: ResolvedOwner =
    scope && typeof scope === 'object'
      ? { connectionId: scope.connectionId ?? null, profile: scope.profile ?? null }
      : { ...ambient, profile: profile ?? ambient.profile }

  const result = await lookup(name)

  if (!result.ok) {
    notify({
      kind: 'error',
      title: translateNow('skills.plugins.deepLinkErrorTitle'),
      message: translateNow(DEEP_LINK_ERROR_KEYS[result.error], name.trim())
    })

    return
  }

  openCatalogPluginInstall(result.entry, profile, owner)
}
