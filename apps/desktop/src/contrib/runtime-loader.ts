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
 * and the two on-disk doors — `<hermes home>/desktop-plugins/<name>/plugin.js`
 * and the unified agent-plugin half `<hermes home>/plugins/<name>/desktop/
 * plugin.js` — the doors the agent writes through.
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

import { installPluginSdk, sdkImportMap } from '@/sdk/runtime'
import { notifyError } from '@/store/notifications'

import { createPluginContext, type HermesPlugin } from './plugin'
import { $pluginRecords, dropPlugin, pluginActive, type PluginKind, publishPlugin } from './plugins-store'

interface LoadOptions {
  /** Root-level default-enable CAP: `false` ships the plugin opt-in (inventory
   *  row, off until the user toggles) even if the plugin says otherwise. The
   *  unified agent-plugin root sets this so `~/.hermes/plugins` keeps its
   *  installed-but-inert posture (GHSA-mcfc-hp25-cjv7) on the desktop side too. */
  defaultEnabled?: boolean
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

    // A disk/runtime copy of a plugin that now ships BUNDLED (e.g. a
    // standalone install of hermes-bots predating its adoption in-tree) must
    // not register a second time: contributions would double up and the two
    // copies would fight over storage. The bundled copy wins; the disk copy
    // is skipped — but VISIBLY: a silent skip left the stale folder
    // undiscoverable while (on shells without the bundled twin) the same
    // folder actively breaks the feature it shadows. The inventory row
    // carries the file path so Settings → Plugins can reveal it for deletion.
    if ($pluginRecords.get()[plugin.id]?.kind === 'bundled') {
      console.info(`[plugins] ${origin} skipped — "${plugin.id}" already ships bundled with the app`)
      publishPlugin({
        id: `${plugin.id}:disk-shadowed`,
        name: `${plugin.name ?? plugin.id} (stale disk copy)`,
        description: `Shadowed by the bundled "${plugin.id}" plugin — this folder is no longer used and can be deleted.`,
        kind: options.kind ?? 'disk',
        file: options.file,
        status: 'disabled'
      })

      return null
    }

    const record = {
      id: plugin.id,
      name: plugin.name ?? plugin.id,
      description: plugin.description,
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
    // reactivates via the handle above) — it just never registers. A root-level
    // `defaultEnabled: false` caps the plugin's own default: the user's explicit
    // enable still wins, a plugin can't self-enable past its root's posture.
    if (pluginActive(plugin.id, (plugin.defaultEnabled ?? true) && (options.defaultEnabled ?? true))) {
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
// The on-disk plugin door — TWO roots, one pipeline:
//  - `<hermes home>/desktop-plugins/<name>/plugin.js` — the standalone door
//    (agent- or user-written desktop-only plugins);
//  - `<hermes home>/plugins/<name>/desktop/plugin.js` — the desktop HALF of a
//    unified agent-plugin package: the same installed folder that carries the
//    Python plugin (plugin.yaml / plugin.json) ships its desktop UI beside it,
//    so one feature is ONE install instead of two co-dependent plugins.
// SELF-MAINTAINING — no reload ceremony:
//  - each plugin.js is fs-watched (the preview watcher IPC, debounced in
//    main): saving the file hot-reloads the plugin in place;
//  - each root is fs-watched too (watchDirectory IPC), so new folders load +
//    removed ones unload on the change tick; older Electron shells without
//    watchDirectory fall back to the slow visible-tab poll.
// Panes land via placement adoption and STAY where the user drags them —
// the tree treats not-yet-loaded pane ids as hidden, so boot and reload are
// collapse -> appear, never a placeholder flash.
// ---------------------------------------------------------------------------

const DISK_POLL_MS = 5_000

interface DiskRoot {
  /** Root-level enable posture, forwarded to the loader (see LoadOptions). */
  defaultEnabled?: boolean
  dir: string
  /** Resolve a scanned folder to its candidate plugin entry file. */
  entry: (folderPath: string) => string
}

/** Both scan roots, resolved fresh each pass (Electron-local, never the
 *  backend's hermes_home — #66899). `agentPluginsRoot` is optional: older
 *  shells predate it and the unified-package half simply doesn't scan. */
async function diskRoots(): Promise<DiskRoot[]> {
  const desktop = window.hermesDesktop

  if (!desktop) {
    return []
  }

  const roots: DiskRoot[] = []
  const standalone = await desktop.desktopPluginsRoot?.()

  if (standalone) {
    roots.push({ dir: standalone, entry: folder => `${folder}/plugin.js` })
  }

  const unified = await desktop.agentPluginsRoot?.()

  if (unified) {
    // Opt-in by default: `~/.hermes/plugins` is installed-but-inert until the
    // user allowlists the Python half (plugins.enabled), so the desktop half
    // matches that posture — inventoried in Settings → Plugins, off until
    // toggled. The standalone desktop-plugins door keeps its default-on trust.
    roots.push({ defaultEnabled: false, dir: unified, entry: folder => `${folder}/desktop/plugin.js` })
  }

  return roots
}

interface DiskPlugin {
  /** Root posture, forwarded on every (re)load of this entry. */
  defaultEnabled?: boolean
  file: string
  /** Loaded plugin id (null while broken — kept so a fixing save reloads). */
  id: null | string
  /** Origin label (folder name) — the toast/inventory name for load errors. */
  origin: string
  watchId: null | string
}

/** Live disk plugins keyed by ENTRY FILE path — unique across both roots
 *  (folder names alone can collide between them). */
const disk = new Map<string, DiskPlugin>()
let watching = false
let scanning = false

/** Drop a folder-named error record — unless that name is the live plugin id
 *  of ANOTHER disk entry (two roots can carry same-named folders; a broken one
 *  must not clobber its healthy namesake's inventory row). */
function dropOriginRecord(origin: string, except: DiskPlugin): void {
  for (const other of disk.values()) {
    if (other !== except && other.id === origin) {
      return
    }
  }

  dropPlugin(origin)
}

async function loadDiskPlugin(entry: DiskPlugin): Promise<void> {
  const desktop = window.hermesDesktop!
  const prevId = entry.id

  try {
    const { text } = await desktop.readFileText(entry.file)

    const id = await loadRuntimePlugin(text, entry.origin, {
      defaultEnabled: entry.defaultEnabled,
      file: entry.file
    })

    // A hot-edit that changes `plugin.id`: loadRuntimePlugin only disposes the
    // NEW id, so unload the previous incarnation here or its contributions +
    // inventory row orphan.
    if (id && prevId && prevId !== id) {
      unloadRuntimePlugin(prevId)
      dropPlugin(prevId)
    }

    entry.id = id ?? entry.id

    // A fixing save under a different plugin id — drop the folder-named
    // error record so the inventory shows one row, not a ghost.
    if (id && id !== entry.origin) {
      dropOriginRecord(entry.origin, entry)
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
    const roots = await diskRoots()

    if (roots.length === 0) {
      return
    }

    const seen = new Set<string>()

    for (const root of roots) {
      let entries

      try {
        ;({ entries } = await desktop.readDir(root.dir))
      } catch {
        continue // Root missing (no plugins yet) — the poll/watch reconciles.
      }

      for (const dir of entries.filter(e => e.isDirectory)) {
        const file = root.entry(dir.path)
        seen.add(file)

        if (disk.has(file)) {
          continue
        }

        try {
          await desktop.readFileText(file)
        } catch {
          continue // No entry file (yet) — not a plugin folder for this root.
        }

        const record: DiskPlugin = {
          defaultEnabled: root.defaultEnabled,
          file,
          id: null,
          origin: dir.name,
          watchId: null
        }

        disk.set(file, record)
        await loadDiskPlugin(record)

        try {
          record.watchId = (await desktop.watchPreviewFile(file)).id
        } catch {
          // Unwatchable — the poll still reconciles new folders; edits need a
          // manual "Reload desktop plugins".
        }
      }
    }

    // Folder deleted -> plugin gone, cleanly (inventory row included).
    for (const [file, record] of disk) {
      if (seen.has(file)) {
        continue
      }

      if (record.id) {
        unloadRuntimePlugin(record.id)
        dropPlugin(record.id)
      }

      dropOriginRecord(record.origin, record)

      if (record.watchId) {
        void desktop.stopPreviewFileWatch(record.watchId)
      }

      disk.delete(file)
    }
  } catch {
    // No plugin roots (or no gateway yet) — nothing to reconcile.
  } finally {
    scanning = false
  }
}

/** Manual rescan (the ⌘K "Reload desktop plugins" fallback). */
export const discoverRuntimePlugins = scanDiskPlugins

/** Start the self-maintaining disk door: initial scan, per-file hot reload,
 *  fs-watched folder reconciliation (poll fallback on older shells). Idempotent. */
export function watchRuntimePlugins(): void {
  const desktop = window.hermesDesktop

  if (watching || !desktop) {
    return
  }

  watching = true

  const dirWatchIds = new Set<string>()
  const watchedDirs = new Set<string>()

  desktop.onPreviewFileChanged(({ id }) => {
    // Directory tick: a plugin folder appeared or vanished — reconcile.
    if (dirWatchIds.has(id)) {
      void scanDiskPlugins()

      return
    }

    for (const record of disk.values()) {
      if (record.watchId === id) {
        void loadDiskPlugin(record)

        return
      }
    }
  })

  // True only when EVERY root is fs-watched — a partially watched set keeps
  // the poll alive so unwatched roots still reconcile new/removed folders.
  const startDirWatches = async (): Promise<boolean> => {
    if (!desktop.watchDirectory) {
      return false
    }

    const roots = await diskRoots()

    if (roots.length === 0) {
      return false
    }

    let all = true

    for (const root of roots) {
      if (watchedDirs.has(root.dir)) {
        continue
      }

      try {
        dirWatchIds.add((await desktop.watchDirectory(root.dir)).id)
        watchedDirs.add(root.dir)
      } catch {
        // Dir missing or unwatchable — the poll covers it and retries here.
        all = false
      }
    }

    return all
  }

  void scanDiskPlugins()
  void startDirWatches().then(watched => {
    if (watched) {
      return
    }

    const timer = window.setInterval(() => {
      if (document.visibilityState !== 'visible') {
        return
      }

      void scanDiskPlugins()

      // A root may have appeared since — upgrade to the watches and retire
      // this poll once every root is covered.
      void startDirWatches().then(upgraded => {
        if (upgraded) {
          window.clearInterval(timer)
        }
      })
    }, DISK_POLL_MS)
  })
}
