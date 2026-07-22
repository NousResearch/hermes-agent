/**
 * Runtime plugin loader — plugins as CODE, not registry edits, loaded after
 * build time. The pipeline every non-bundled plugin takes:
 *
 *   source (plain ESM js) -> [integrity check] -> bare-specifier rewrite
 *   (`@hermes/plugin-sdk` / `react*` -> live shim blobs, see sdk/runtime.ts)
 *   -> blob `import()` -> validate default HermesPlugin -> register(ctx)
 *
 * Loading the same plugin id again disposes the previous registrations first
 * (agent rewrites a plugin file -> clean reload). Failures toast + log; a
 * broken plugin can never take the app down.
 *
 * Sources today: the in-repo runtime example (`?raw`, proves the pipeline)
 * and `<hermes home>/desktop-plugins/<name>/plugin.js` on disk — the door the
 * agent writes through.
 *
 * SECURITY — this is NOT a capability boundary. A loaded plugin is evaluated
 * as ESM in the renderer realm with FULL app authority: the React singleton,
 * the whole SDK (`host.request` gateway RPC, `ctx.rest`, storage, `navigate`).
 * The isolation here is *error* isolation only (ContribBoundary, isolated
 * listeners) — a plugin can't crash the app, but it can do anything the app
 * can. That's acceptable for local sources (disk files can already run code),
 * and `integrity` only proves the bytes match a hash — it does NOT sandbox.
 * A remote source (https + allowlist) must NOT reuse this pipeline as-is:
 * it needs a real boundary (iframe/worker + CSP + capability gating) before
 * it can land. The `{ integrity }` option is the transport seam, not the
 * trust seam.
 */

import { getStatus } from '@/hermes'
import { installPluginSdk, sdkImportMap } from '@/sdk/runtime'
import { notifyError } from '@/store/notifications'

import { createPluginContext, type HermesPlugin } from './plugin'
import { dropPlugin, pluginActive, type PluginKind, publishPlugin } from './plugins-store'

interface LoadOptions {
  /** Absolute plugin.js path (disk plugins) — recorded for reveal/inventory. */
  file?: string
  /** `sha256-<base64>` — verified against the source before evaluation. */
  integrity?: string
  /** Inventory bucket; the disk door is the default runtime source. */
  kind?: PluginKind
}

/** Live runtime plugins: id -> disposers (unload/reload support). */
const loaded = new Map<string, (() => void)[]>()

// Matches the specifier of a static `from '…'`, a side-effect `import '…'`, or
// a dynamic `import('…')` — anchored to import/export syntax so a bare string
// literal or comment (e.g. `notify('react')`) is never touched.
const importSpecifierRe = () => /(from\s*|import\s*\(\s*|import\s+)(['"])([^'"]+)\2/g

/** Rewrite ONLY mapped import specifiers (@hermes/plugin-sdk, react*) to their
 *  live shim blob URLs — never occurrences inside strings/comments. */
function rewriteSpecifiers(source: string): string {
  const map = sdkImportMap()

  return source.replace(importSpecifierRe(), (whole, pre, quote, spec) =>
    map[spec] ? `${pre}${quote}${map[spec]}${quote}` : whole
  )
}

/** Bare import specifiers the loader can't resolve (not relative/URL, not in
 *  the SDK map). Surfaced up-front so they don't fail as a cryptic native
 *  "Failed to resolve module specifier" from the blob import. */
function unsupportedImports(source: string): string[] {
  const map = sdkImportMap()
  const bare = new Set<string>()

  for (const m of source.matchAll(importSpecifierRe())) {
    const spec = m[3]

    // Skip relative/absolute (./ ../ /) and any URL scheme (blob: http(s):).
    if (spec && !/^[./]/.test(spec) && !/^[a-z][a-z0-9+.-]*:/i.test(spec) && !map[spec]) {
      bare.add(spec)
    }
  }

  return [...bare]
}

async function verifyIntegrity(source: string, integrity: string): Promise<boolean> {
  const [algo, expected] = integrity.split('-', 2)

  if (algo !== 'sha256' || !expected) {
    return false
  }

  const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(source))
  // Standard SRI base64 (`sha256-<base64>`) — a base64url-encoded hash won't match.
  const actual = btoa(String.fromCharCode(...new Uint8Array(digest)))

  return actual === expected
}

export function unloadRuntimePlugin(id: string): void {
  loaded.get(id)?.forEach(dispose => dispose())
  loaded.delete(id)
}

/** Evaluate + register one runtime plugin. Returns its id, or null on failure. */
export async function loadRuntimePlugin(
  source: string,
  origin: string,
  options: LoadOptions = {}
): Promise<null | string> {
  installPluginSdk()

  try {
    if (options.integrity && !(await verifyIntegrity(source, options.integrity))) {
      throw new Error(`integrity check failed for ${origin}`)
    }

    const unsupported = unsupportedImports(source)

    if (unsupported.length > 0) {
      throw new Error(
        `unsupported import${unsupported.length > 1 ? 's' : ''}: ${unsupported.join(', ')} — ` +
          `runtime plugins may only import @hermes/plugin-sdk and react`
      )
    }

    const url = URL.createObjectURL(new Blob([rewriteSpecifiers(source)], { type: 'text/javascript' }))

    let mod: { default?: HermesPlugin }

    try {
      mod = await import(/* @vite-ignore */ url)
    } finally {
      URL.revokeObjectURL(url)
    }

    const plugin = mod.default

    if (!plugin?.id || typeof plugin.register !== 'function') {
      throw new Error(`${origin} has no valid default HermesPlugin export`)
    }

    const record = {
      id: plugin.id,
      name: plugin.name ?? plugin.id,
      kind: options.kind ?? 'disk',
      file: options.file
    }

    const activate = () => {
      // Reload = dispose the previous incarnation, then register fresh.
      unloadRuntimePlugin(plugin.id)
      const disposers: (() => void)[] = []
      plugin.register(createPluginContext(plugin.id, dispose => disposers.push(dispose)))
      loaded.set(plugin.id, disposers)
      publishPlugin({ ...record, status: 'loaded' })
    }

    publishPlugin({ ...record, status: 'disabled' }, { activate, deactivate: () => unloadRuntimePlugin(plugin.id) })

    // A disabled plugin still inventories (settings shows it, toggle
    // reactivates via the handle above) — it just never registers.
    if (pluginActive(plugin.id, plugin.defaultEnabled ?? true)) {
      activate()
    }

    return plugin.id
  } catch (error) {
    console.error(`[plugins] runtime load failed (${origin})`, error)
    notifyError(error, `Plugin "${origin}" failed to load`)
    publishPlugin({
      id: origin,
      name: origin,
      kind: options.kind ?? 'disk',
      file: options.file,
      status: 'error',
      error: error instanceof Error ? error.message : String(error)
    })

    return null
  }
}

// ---------------------------------------------------------------------------
// The on-disk plugin door: `<hermes home>/desktop-plugins/<name>/plugin.js`
// (agent- or user-written). SELF-MAINTAINING — no reload ceremony:
//  - each plugin.js is fs-watched (the preview watcher IPC, debounced in
//    main): saving the file hot-reloads the plugin in place;
//  - a slow visible-tab poll of the directory picks up new folders (load +
//    watch) and removed ones (unload + unwatch).
// Panes land via placement adoption and STAY where the user drags them —
// the tree treats not-yet-loaded pane ids as hidden, so boot and reload are
// collapse -> appear, never a placeholder flash.
// ---------------------------------------------------------------------------

const DISK_POLL_MS = 5_000

interface DiskPlugin {
  file: string
  /** Loaded plugin id (null while broken — kept so a fixing save reloads). */
  id: null | string
  watchId: null | string
}

const disk = new Map<string, DiskPlugin>()
let watching = false
let scanning = false

async function loadDiskPlugin(name: string, file: string): Promise<void> {
  const desktop = window.hermesDesktop!
  const entry = disk.get(name)
  const prevId = entry?.id

  try {
    const { text } = await desktop.readFileText(file)
    const id = await loadRuntimePlugin(text, name, { file })

    // A hot-edit that changes `plugin.id`: loadRuntimePlugin only disposes the
    // NEW id, so unload the previous incarnation here or its contributions +
    // inventory row orphan.
    if (id && prevId && prevId !== id) {
      unloadRuntimePlugin(prevId)
      dropPlugin(prevId)
    }

    if (entry) {
      entry.id = id ?? entry.id
    }

    // A fixing save under a different plugin id — drop the folder-named
    // error record so the inventory shows one row, not a ghost.
    if (id && id !== name) {
      dropPlugin(name)
    }
  } catch {
    // File vanished mid-read — the next scan reconciles.
  }
}

async function scanDiskPlugins(): Promise<void> {
  const desktop = window.hermesDesktop

  // Re-entrancy guard: the 5s poll must not overlap a slow in-flight scan
  // (reads/loads can exceed the interval).
  if (!desktop || scanning) {
    return
  }

  scanning = true

  try {
    const { hermes_home } = await getStatus()
    const { entries } = await desktop.readDir(`${hermes_home}/desktop-plugins`)
    const seen = new Set<string>()

    for (const dir of entries.filter(e => e.isDirectory)) {
      seen.add(dir.name)

      if (disk.has(dir.name)) {
        continue
      }

      const file = `${dir.path}/plugin.js`

      try {
        await desktop.readFileText(file)
      } catch {
        continue // No plugin.js (yet) — not a plugin folder.
      }

      const record: DiskPlugin = { file, id: null, watchId: null }
      disk.set(dir.name, record)
      await loadDiskPlugin(dir.name, file)

      try {
        record.watchId = (await desktop.watchPreviewFile(file)).id
      } catch {
        // Unwatchable — the poll still reconciles new folders; edits need a
        // manual "Reload desktop plugins".
      }
    }

    // Folder deleted -> plugin gone, cleanly (inventory row included).
    for (const [name, record] of disk) {
      if (seen.has(name)) {
        continue
      }

      if (record.id) {
        unloadRuntimePlugin(record.id)
        dropPlugin(record.id)
      }

      dropPlugin(name)

      if (record.watchId) {
        void desktop.stopPreviewFileWatch(record.watchId)
      }

      disk.delete(name)
    }
  } catch {
    // No desktop-plugins dir (or no gateway yet) — nothing to reconcile.
  } finally {
    scanning = false
  }
}

/** Manual rescan (the ⌘K "Reload desktop plugins" fallback). */
export const discoverRuntimePlugins = scanDiskPlugins

/** Start the self-maintaining disk door: initial scan, per-file hot reload,
 *  slow folder reconciliation while the window is visible. Idempotent. */
export function watchRuntimePlugins(): void {
  const desktop = window.hermesDesktop

  if (watching || !desktop) {
    return
  }

  watching = true

  desktop.onPreviewFileChanged(({ id }) => {
    for (const [name, record] of disk) {
      if (record.watchId === id) {
        void loadDiskPlugin(name, record.file)

        return
      }
    }
  })

  void scanDiskPlugins()
  window.setInterval(() => {
    if (document.visibilityState === 'visible') {
      void scanDiskPlugins()
    }
  }, DISK_POLL_MS)
}
