import type { PluginRecord } from '@/contrib/plugins-store'
import type { AgentPluginRow } from '@/store/agent-plugins'

/**
 * One row per PACKAGE on the Plugins page. A package may have a desktop half
 * (loaded into this app, same for every profile), an agent half (installed in
 * the SELECTED profile's backend), or both — a unified package whose folder
 * ships `plugin.yaml` next to `desktop/plugin.js`.
 *
 * The join key is the package folder name: the agent row's `name` and the
 * desktop record's `packageName` (stamped by Electron when it copies the half
 * out to the app root). A standalone desktop plugin or an agent-only package
 * simply has one side empty.
 */
export type PackageKind = 'agent' | 'both' | 'desktop'

export interface PluginPackage {
  key: string
  name: string
  description: string
  kind: PackageKind
  desktop: null | PluginRecord
  agent: AgentPluginRow | null
  /** Agent half exists on disk somewhere (its desktop half is here) but is not
   *  installed in the selected profile — the "Install here" affordance. */
  agentMissingInProfile: boolean
  /** Desktop half is declared by the agent package but the app has no copy yet
   *  (Electron's reconcile has not run, or the copy failed). */
  desktopMissing: boolean
}

/** Folder name when a desktop record lives in the UNIFIED agent-plugins root
 *  (`~/.hermes/plugins/<name>/desktop/plugin.js`) — legacy records from before
 *  halves were copied to the app root. */
function legacyPackageName(file?: string): null | string {
  if (!file) {
    return null
  }

  const match = /[\\/]plugins[\\/]([^\\/]+)[\\/]desktop[\\/]plugin\.js$/.exec(file)

  return match ? match[1] : null
}

export function desktopPackageName(record: PluginRecord): null | string {
  return record.packageName ?? legacyPackageName(record.file)
}

const KIND_RANK: Record<PackageKind, number> = { both: 0, agent: 1, desktop: 2 }
const DESKTOP_KIND_RANK: Record<PluginRecord['kind'], number> = { disk: 0, runtime: 1, bundled: 2 }

export function mergePluginPackages(
  desktopRecords: readonly PluginRecord[],
  agentRows: readonly AgentPluginRow[]
): PluginPackage[] {
  const byKey = new Map<string, PluginPackage>()

  for (const row of agentRows) {
    const key = row.name
    byKey.set(key, {
      key,
      name: row.name,
      description: row.description,
      kind: row.has_desktop_half ? 'both' : 'agent',
      desktop: null,
      agent: row,
      agentMissingInProfile: false,
      desktopMissing: Boolean(row.has_desktop_half)
    })
  }

  for (const record of desktopRecords) {
    const pkg = desktopPackageName(record)

    if (pkg) {
      const existing = byKey.get(pkg)

      if (existing) {
        existing.desktop = record
        existing.kind = 'both'
        existing.desktopMissing = false
        // The desktop half carries the human display name ("Pixel Overlay"
        // vs the folder-shaped agent name); keep the title stable whichever
        // profile is selected.
        existing.name = record.name
        existing.description ||= record.description ?? ''

        continue
      }

      // Desktop half present, agent half absent from THIS profile.
      byKey.set(pkg, {
        key: pkg,
        name: record.name,
        description: record.description ?? '',
        kind: 'both',
        desktop: record,
        agent: null,
        agentMissingInProfile: true,
        desktopMissing: false
      })

      continue
    }

    byKey.set(`desktop:${record.id}`, {
      key: `desktop:${record.id}`,
      name: record.name,
      description: record.description ?? '',
      kind: 'desktop',
      desktop: record,
      agent: null,
      agentMissingInProfile: false,
      desktopMissing: false
    })
  }

  return [...byKey.values()].sort((a, b) => {
    const rank = KIND_RANK[a.kind] - KIND_RANK[b.kind]

    if (rank !== 0) {
      return rank
    }

    if (a.desktop && b.desktop) {
      const dk = DESKTOP_KIND_RANK[a.desktop.kind] - DESKTOP_KIND_RANK[b.desktop.kind]

      if (dk !== 0) {
        return dk
      }
    }

    return a.name.localeCompare(b.name)
  })
}
