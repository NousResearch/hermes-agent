import type { ReadableAtom } from 'nanostores'

import type { PluginRecord } from '@/contrib/plugins-store'
import { resolveDiskPluginEntry } from '@/contrib/runtime-loader'

/**
 * Cheap pre-check: could a hybrid repo's desktop half be served by the unified
 * `plugins/<name>/desktop/plugin.js` the agent install lands, instead of a
 * second copy under `desktop-plugins/<name>/`? Two copies load the same plugin
 * id twice (#100412).
 *
 * This only rules the case IN; the install itself skips the copy on hard
 * evidence (`findUnifiedDesktopEntry`). Everything here fails toward copying:
 * a remote backend's `plugins/` tree is not on this machine, an unknown mode
 * proves nothing, the unified door only serves the nested `desktop/plugin.js`
 * shape (a root-level `plugin.js` would never load), and an existing
 * standalone copy is the one the loader serves — Force must keep refreshing
 * it rather than leave it stale behind a newer unified half.
 */
export interface HybridInstallPlanInput {
  connectionMode: 'local' | 'remote' | undefined
  probeAgent: boolean
  probeDesktop: boolean
  /** Where the repo keeps its desktop half: '.' (root plugin.js) or 'desktop'. */
  desktopSourceSubdir: '.' | 'desktop' | null
  /** `desktop-plugins/<name>/plugin.js` already exists from an earlier install. */
  standaloneCopy: boolean
  installAgent: boolean
  installDesktop: boolean
}

export function desktopHalfMayShareLocalRoot(input: HybridInstallPlanInput): boolean {
  return (
    input.connectionMode === 'local' &&
    input.probeAgent &&
    input.probeDesktop &&
    input.desktopSourceSubdir === 'desktop' &&
    !input.standaloneCopy &&
    input.installAgent &&
    input.installDesktop
  )
}

/** The slice of the Electron bridge the entry walks need (injectable for tests). */
export type PluginRootsFs = Pick<NonNullable<Window['hermesDesktop']>, 'readDir'> &
  Partial<Pick<NonNullable<Window['hermesDesktop']>, 'agentPluginsRoot' | 'desktopPluginsRoot'>>

async function resolveEntryUnder(
  desktop: PluginRootsFs | undefined,
  root: 'agentPluginsRoot' | 'desktopPluginsRoot',
  segments: readonly string[]
): Promise<null | string> {
  const rootFn = desktop?.[root]

  if (!desktop || !rootFn) {
    return null
  }

  try {
    return await resolveDiskPluginEntry(desktop as Window['hermesDesktop'], await rootFn(), segments)
  } catch {
    return null // Root unreadable / IPC failure — no evidence, so copy.
  }
}

/**
 * Hard evidence for skipping the copy: resolve `<agentPluginsRoot>/<name>/
 * desktop/plugin.js` with the loader's own entry resolver, so the answer is
 * exactly "would the disk door pick this up" and the two can't drift. Returns
 * the entry's absolute path (as the loader records it) or null.
 */
export function findUnifiedDesktopEntry(
  desktop: PluginRootsFs | undefined,
  pluginName: string
): Promise<null | string> {
  return resolveEntryUnder(desktop, 'agentPluginsRoot', [pluginName, 'desktop', 'plugin.js'])
}

/**
 * An earlier install's `<desktopPluginsRoot>/<desktopName>/plugin.js` (the
 * folder `installDesktopPlugin` writes). While it exists the loader serves it,
 * so the install must keep going through `installDesktopPlugin` — Force
 * refreshes that copy, no-Force reports it as already installed.
 */
export function findStandaloneDesktopEntry(
  desktop: PluginRootsFs | undefined,
  desktopName: string
): Promise<null | string> {
  return resolveEntryUnder(desktop, 'desktopPluginsRoot', [desktopName, 'plugin.js'])
}

/** Rescan-and-wait rounds for the unified record. A rescan is dropped while
 *  another scan holds the loader's lock, and fully watched roots have no
 *  fallback poll to catch up, so one round is not enough; three rounds of 2 s
 *  outlast any in-flight scan without holding the modal open for long. In the
 *  happy path the record is already published when round 1 looks. */
export const UNIFIED_RECORD_ATTEMPTS = 3
export const UNIFIED_RECORD_WAIT_MS = 2_000

/** What the disk door said about the entry file: a live record to enable, or
 *  a terminal reason it never will be (failed to load, shadowed by a bundled
 *  copy). */
export type UnifiedRecordOutcome = { id: string } | { error: string }

/**
 * The unified half loads OPT-IN (`defaultEnabled: false` on the `plugins/`
 * root). Once the disk door publishes the record for the entry file, the
 * modal can honour the ticked "Desktop UI" box the way `enable` honours the
 * agent half. Null when no round produced any record — the plugin then stays
 * opt-in.
 */
export async function settleUnifiedDesktopPluginId(
  rescan: () => Promise<void>,
  records: ReadableAtom<Record<string, PluginRecord>>,
  entryFile: string,
  attempts = UNIFIED_RECORD_ATTEMPTS,
  waitMs = UNIFIED_RECORD_WAIT_MS
): Promise<null | UnifiedRecordOutcome> {
  for (let round = 0; round < attempts; round += 1) {
    await rescan()

    const outcome = await waitForUnifiedDesktopPluginId(records, entryFile, waitMs)

    if (outcome) {
      return outcome
    }
  }

  return null
}

/** Resolve as soon as the disk door publishes anything for the entry file, or null after `timeoutMs`. */
export function waitForUnifiedDesktopPluginId(
  records: ReadableAtom<Record<string, PluginRecord>>,
  entryFile: string,
  timeoutMs: number
): Promise<null | UnifiedRecordOutcome> {
  return new Promise(resolve => {
    let settled = false
    let unsubscribe: (() => void) | null = null

    const timer = setTimeout(() => finish(null), timeoutMs)

    function finish(outcome: null | UnifiedRecordOutcome) {
      if (settled) {
        return
      }

      settled = true
      clearTimeout(timer)
      unsubscribe?.()
      resolve(outcome)
    }

    // `subscribe` fires with the current value BEFORE returning the
    // unsubscriber, so an already-published record settles here and the
    // listener is released right after.
    unsubscribe = records.subscribe(current => {
      const outcome = findUnifiedDesktopPluginId(current, entryFile)

      if (outcome) {
        finish(outcome)
      }
    })

    if (settled) {
      unsubscribe()
    }
  })
}

const SHADOWED_SUFFIX = ':disk-shadowed'

export function findUnifiedDesktopPluginId(
  records: Record<string, PluginRecord>,
  entryFile: string
): null | UnifiedRecordOutcome {
  for (const record of Object.values(records)) {
    if (record.kind !== 'disk' || record.file !== entryFile) {
      continue
    }

    // The loader's "a bundled copy of this id already ships" inventory row.
    if (record.id.endsWith(SHADOWED_SUFFIX)) {
      return { error: record.description ?? record.name }
    }

    if (record.status === 'error') {
      return { error: record.error ?? record.name }
    }

    return { id: record.id }
  }

  return null
}
