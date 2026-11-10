/**
 * PLUGIN INVENTORY — the reactive record of every desktop plugin the app
 * knows about (bundled `src/plugins/*`, the in-repo runtime example, the
 * `<hermes home>/desktop-plugins/*` disk door — incl. agent-written ones),
 * plus the persisted DISABLED set. The settings "Plugins" page renders this;
 * the loaders publish into it and consult the disabled set before
 * registering. Enable/disable is live: each record carries the loader's own
 * activate/deactivate handles, so toggling never needs an app reload.
 */

import { atom } from 'nanostores'

export type PluginKind = 'bundled' | 'disk' | 'runtime'
export type PluginStatus = 'disabled' | 'error' | 'loaded'

export interface PluginRecord {
  id: string
  name: string
  kind: PluginKind
  status: PluginStatus
  /** One-liner from the plugin's own metadata (what it adds). */
  description?: string
  /** Sensitive doors the plugin declared (`HermesPlugin.capabilities`). */
  capabilities?: string[]
  /** Load/registration failure message (status 'error'). */
  error?: string
  /** Absolute plugin.js path (disk plugins) — powers "Reveal in Finder". */
  file?: string
  /** Agent package this is the desktop half of (unified agent+desktop packages). */
  packageName?: string
  /** Where that package came from (catalog sidecar or git remote), when known. */
  packageOrigin?: PackageOrigin
}

/** Where a disk plugin came from: a package half's marker origin, or the catalog entry a standalone plugin's
 *  installer recorded (`pin` is the tag object a tag pin names, beside the commit `sha` checked out). */
export interface PackageOrigin {
  catalogName?: string
  pin?: string
  repo?: string
  sha?: string
}

// Explicit user enable/disable choices, id -> boolean. ABSENCE means "no
// choice" — the plugin falls back to its own `defaultEnabled`. This is what
// lets an opt-in plugin ship off-by-default: absence ≠ enabled anymore.
const DECISIONS_KEY = 'hermes.desktop.pluginDecisions.v2'
const LEGACY_DISABLED_KEY = 'hermes.desktop.disabledPlugins.v1'

function loadDecisions(): Record<string, boolean> {
  try {
    const raw = window.localStorage.getItem(DECISIONS_KEY)

    if (raw) {
      return JSON.parse(raw) as Record<string, boolean>
    }

    // Migrate the v1 disabled-set: each disabled id becomes an explicit `false`.
    const legacy = window.localStorage.getItem(LEGACY_DISABLED_KEY)

    if (legacy) {
      return Object.fromEntries((JSON.parse(legacy) as string[]).map(id => [id, false]))
    }
  } catch {
    // Nonfatal — fall through to no choices.
  }

  return {}
}

export const $pluginDecisions = atom<Record<string, boolean>>(loadDecisions())

/** Whether a plugin should register: the user's explicit choice if any, else
 *  the plugin's own default (true for ordinary plugins, false for opt-in). */
export function pluginActive(id: string, defaultEnabled = true): boolean {
  const decisions = $pluginDecisions.get()

  return id in decisions ? decisions[id] : defaultEnabled
}

function saveDecisions(next: Record<string, boolean>) {
  $pluginDecisions.set(next)

  try {
    window.localStorage.setItem(DECISIONS_KEY, JSON.stringify(next))
  } catch {
    // Nonfatal.
  }
}

export const $pluginRecords = atom<Record<string, PluginRecord>>({})

/** Loader-owned lifecycle controls for a plugin (activate/deactivate). */
interface PluginHandle {
  activate: () => Promise<void> | void
  deactivate: () => void
}

/** Loader-owned lifecycle handles, keyed by plugin id. */
const handles = new Map<string, PluginHandle>()

/** Publish/refresh a plugin's record + its activate/deactivate handles. */
export function publishPlugin(record: PluginRecord, handle?: PluginHandle): void {
  $pluginRecords.set({ ...$pluginRecords.get(), [record.id]: record })

  if (handle) {
    handles.set(record.id, handle)
  }
}

export function patchPlugin(id: string, patch: Partial<PluginRecord>): void {
  const current = $pluginRecords.get()[id]

  if (current) {
    $pluginRecords.set({ ...$pluginRecords.get(), [id]: { ...current, ...patch } })
  }
}

export function dropPlugin(id: string): void {
  const { [id]: _dropped, ...rest } = $pluginRecords.get()
  $pluginRecords.set(rest)
  handles.delete(id)
}

/** Turn on the desktop half of unified package *packageName*. A half ships
 *  opt-in to match its inert agent half, so turning the package on (the
 *  install dialog's "Enable after install", the hub's Agent switch) turns the
 *  half on too instead of leaving a second switch to find. *keepUserChoice*
 *  leaves a half the user explicitly switched off alone. */
export async function enablePackageDesktopHalf(packageName: string, { keepUserChoice = false } = {}): Promise<void> {
  await enableDesktopPlugins(record => record.packageName === packageName, keepUserChoice)
}

/** The catalog entry a STANDALONE plugin was installed from; a package's half answers for no entry here. */
const standaloneCatalogName = (record: PluginRecord) =>
  record.packageName ? undefined : record.packageOrigin?.catalogName

/** The catalog entries the STANDALONE desktop plugins on disk were installed from, each with the commits
 *  that count as current for it (the one checked out, and the tag object a tag pin names). A desktop-only
 *  entry has no agent row to carry its `catalog_name`, so a surface asking "is this entry installed?" reads
 *  these too. A package's desktop half is left out: desktop plugins are app-wide, but whether a package is
 *  installed is the selected profile's agent inventory. */
export function installedDesktopCatalogEntries(records: Record<string, PluginRecord>): Map<string, string[]> {
  return new Map(
    Object.values(records).flatMap(record => {
      const name = standaloneCatalogName(record)
      const { pin, sha } = record.packageOrigin ?? {}

      return name ? [[name, [sha, pin].filter((value): value is string => Boolean(value))] as const] : []
    })
  )
}

/** Turn on the standalone desktop plugin installed from catalog entry *catalogName*. A plugin can
 *  ship off by default (`defaultEnabled: false`); a dialog that said "turns on right away" must not
 *  leave it off for the user to find. Ticking that is the user's choice, so an older "off" yields. */
export async function enableCatalogDesktopPlugin(catalogName: string): Promise<void> {
  await enableDesktopPlugins(record => standaloneCatalogName(record) === catalogName, false)
}

async function enableDesktopPlugins(matches: (record: PluginRecord) => boolean, keepUserChoice: boolean) {
  const decisions = $pluginDecisions.get()

  for (const record of Object.values($pluginRecords.get())) {
    if (!matches(record) || record.status !== 'disabled') {
      continue
    }

    if (!keepUserChoice || !(record.id in decisions)) {
      await setPluginEnabled(record.id, true)
    }
  }
}

/** Live toggle: deactivate + remember, or forget + reactivate. */
export async function setPluginEnabled(id: string, enabled: boolean): Promise<void> {
  saveDecisions({ ...$pluginDecisions.get(), [id]: enabled })

  const handle = handles.get(id)

  if (!handle) {
    return
  }

  if (enabled) {
    await handle.activate()
  } else {
    handle.deactivate()
    patchPlugin(id, { status: 'disabled' })
  }
}
