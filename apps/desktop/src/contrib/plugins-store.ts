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
  /** Load/registration failure message (status 'error'). */
  error?: string
  /** Absolute plugin.js path (disk plugins) — powers "Reveal in Finder". */
  file?: string
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
