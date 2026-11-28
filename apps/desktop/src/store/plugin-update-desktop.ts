import type { PluginsManageResult } from '@hermes/shared'

import { $pluginRecords } from '@/contrib/plugins-store'
import { discoverRuntimePlugins } from '@/contrib/runtime-loader'
import { isBrowserHostedDesktop } from '@/lib/platform'

type Provenance = Pick<PluginsManageResult, 'has_desktop_half' | 'installed_repo' | 'installed_subdir' | 'installed_sha'>

/** Only the update reply can identify the newly installed tree. */
export async function refreshUpdatedDesktopHalf(
  provenance: Provenance,
  remote: boolean,
  fallbackHalf?: boolean,
  catalogName?: string
): Promise<boolean> {
  if (!remote || isBrowserHostedDesktop()) {
    return true
  }

  if (provenance.has_desktop_half === false) {
    // A removed half leaves the old app-level copy behind; do not call that fully updated.
    return fallbackHalf !== true
  }

  if (provenance.has_desktop_half == null && fallbackHalf === false) {
    return true
  }

  const { installed_repo: repo, installed_subdir: subdir, installed_sha: sha } = provenance
  const install = window.hermesDesktop?.installDesktopPlugin

  if (!repo || !sha || !/^[0-9a-f]{40}$/i.test(sha) || !install) {
    return false
  }

  try {
    const result = await install({ identifier: subdir ? `${repo}#${subdir}` : repo, force: true, sha, catalogName })

    if (!result.ok) {
      return false
    }

    await discoverRuntimePlugins()

    return Boolean(result.pluginName && Object.values($pluginRecords.get()).some(record =>
      record.packageName === result.pluginName &&
      (record.status === 'loaded' || record.status === 'disabled') &&
      !record.error && record.packageOrigin?.sha?.toLowerCase() === sha.toLowerCase()
    ))
  } catch {
    return false
  }
}
