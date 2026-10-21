/**
 * MAIN-process discovery of remote gateway plugins dropped into the
 * desktop plugin tree. RemotePlugins run in the main process because they own
 * sockets; ordinary desktop plugins (`plugin.js`) run in the renderer.
 *
 * Contract:
 *  - A plugin lives at `<desktop-plugins-root>/<name>/remote.mjs`
 *    (or `remote.js`) and default-exports a `RemotePlugin`-shaped
 *    object. The renderer half (`plugin.js`) is optional — a folder with
 *    only `remote.mjs` works and surfaces `scheme` + `hint` through the
 *    generic Add-Connection UI.
 *  - Validation happens here, once, at load: id + RFC3986 scheme + callable
 *    `connect`; `http`/`https` are refused; duplicate scheme claims are
 *    first-wins with a warning. A folder that throws at import is skipped —
 *    one bad plugin must not brick the app's connection layer.
 *  - Enable/disable reuses the persisted renderer-plugin state
 *    (`pluginActive`, Settings → Plugins): one row per plugin id controls
 *    both halves.
 *
 * RemotePlugins are `import()`ed in the MAIN process: plugin code runs with
 * main-process privileges.
 */
import fs from 'node:fs'
import path from 'node:path'
import { pathToFileURL } from 'node:url'

import { canonicalScheme, type RemotePlugin } from './remote-plugin-registry'

export const REMOTE_PLUGIN_MODULE_BASENAMES = ['remote.mjs', 'remote.js'] as const

const BUILT_IN_SCHEMES = new Set(['http', 'https'])

export interface RemotePluginLoadIssue {
  /** Plugin folder name (stable even when the module never loaded). */
  folder: string
  /** What we tried (module path) or the plugin id when validation got that far. */
  detail: string
}

export interface RemotePluginLoadResult {
  plugins: RemotePlugin[]
  /** Non-fatal per-folder problems, surfaced through the main-process log. */
  issues: RemotePluginLoadIssue[]
}

/** Validate an imported module's default export. Returns null (with a
 * human-readable reason) for anything unusable. */
export function validateRemotePluginModule(mod: unknown): { plugin: null | RemotePlugin; reason: null | string } {
  const plugin = (mod as { default?: unknown })?.default as Partial<RemotePlugin> | undefined

  if (!plugin || typeof plugin !== 'object') {
    return { plugin: null, reason: 'no default export' }
  }

  if (typeof plugin.id !== 'string' || !plugin.id.trim()) {
    return { plugin: null, reason: 'missing id' }
  }

  const scheme = canonicalScheme(plugin.scheme)

  if (!scheme) {
    return { plugin: null, reason: `invalid scheme '${String(plugin.scheme)}'` }
  }

  if (BUILT_IN_SCHEMES.has(scheme)) {
    return { plugin: null, reason: `'${scheme}' is a built-in scheme and is not registrable` }
  }

  if (typeof plugin.connect !== 'function') {
    return { plugin: null, reason: 'connect is not a function' }
  }

  if (plugin.claims !== undefined && typeof plugin.claims !== 'function') {
    return { plugin: null, reason: 'claims must be a function when present' }
  }

  if (plugin.normalize !== undefined && typeof plugin.normalize !== 'function') {
    return { plugin: null, reason: 'normalize must be a function when present' }
  }

  if (plugin.pastePattern !== undefined) {
    if (typeof plugin.pastePattern !== 'string') {
      return { plugin: null, reason: 'pastePattern must be a string when present' }
    }

    try {
      new RegExp(plugin.pastePattern)
    } catch {
      return { plugin: null, reason: 'pastePattern is not a valid regular expression' }
    }
  }

  return { plugin: { ...plugin, scheme } as RemotePlugin, reason: null }
}

/**
 * Discover + load every enabled plugin under `root`.
 *
 * @param root the desktop-plugins root (from desktop-plugins-root.ts)
 * @param isEnabled plugin-enable predicate (defaults: enabled)
 * @param dynamicImport injectable import() (tests fake the module system)
 */
export async function loadRemotePlugins({
  root,
  isEnabled = () => true,
  dynamicImport = (specifier: string) => import(specifier),
  log = () => {}
}: {
  root: string
  isEnabled?: (pluginId: string) => boolean | Promise<boolean>
  dynamicImport?: (specifier: string) => Promise<unknown>
  log?: (msg: string) => void
}): Promise<RemotePluginLoadResult> {
  const plugins: RemotePlugin[] = []
  const issues: RemotePluginLoadIssue[] = []
  const claimed = new Map<string, string>()

  let entries: fs.Dirent[] = []

  try {
    entries = await fs.promises.readdir(root, { withFileTypes: true })
  } catch {
    // No desktop-plugins root yet — zero plugins is the normal state.
    return { plugins, issues }
  }

  for (const entry of entries) {
    if (!entry.isDirectory()) {
      continue
    }

    const folder = entry.name

    const modulePath = REMOTE_PLUGIN_MODULE_BASENAMES
      .map(basename => path.join(root, folder, basename))
      .find(candidate => fs.existsSync(candidate))

    if (!modulePath) {
      continue
    }

    if (!(await isEnabled(folder))) {
      log(`[remote-plugins] '${folder}' is disabled — plugin not loaded`)

      continue
    }

    let mod: unknown

    try {
      mod = await dynamicImport(pathToFileURL(modulePath).href)
    } catch (err) {
      const reason = String((err as Error)?.message ?? err)

      log(`[remote-plugins] '${folder}' failed to import: ${reason}`)
      issues.push({ folder, detail: reason })

      continue
    }

    const { plugin, reason } = validateRemotePluginModule(mod)

    if (!plugin) {
      log(`[remote-plugins] '${folder}' rejected: ${reason}`)
      issues.push({ folder, detail: reason ?? 'invalid plugin' })

      continue
    }

    const existing = claimed.get(plugin.scheme)

    if (existing) {
      log(`[remote-plugins] '${folder}' claims '${plugin.scheme}://' already owned by '${existing}' — ignored`)
      issues.push({ folder, detail: `duplicate scheme '${plugin.scheme}' (owned by '${existing}')` })

      continue
    }

    claimed.set(plugin.scheme, folder)
    plugins.push(plugin)
    log(`[remote-plugins] loaded '${plugin.id}' claiming '${plugin.scheme}://'`)
  }

  return { plugins, issues }
}
