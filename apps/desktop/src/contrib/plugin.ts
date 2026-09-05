/**
 * The plugin authoring contract. A plugin is a file that default-exports a
 * `HermesPlugin`; it never touches the registry directly — it receives a
 * scoped `PluginContext` whose `register` auto-tags provenance
 * (`source: 'plugin:<id>'`) and namespaces the contribution id
 * (`<id>:<localId>`), so authors write plain contributions and collisions
 * between plugins are impossible.
 *
 * Bundled plugins live in `src/plugins/<name>/plugin.tsx` and are discovered
 * by `discoverBundledPlugins()` (contrib/plugins.ts) — no import, no registry
 * edit. Runtime-fetched third-party plugins will drive the SAME contract
 * through the plugin host loader (next phase); this is that seam.
 */

import { pluginRest, type PluginRestOptions, pluginSocket } from '@/hermes'
import { createPluginI18n, type PluginI18n } from '@/i18n'
import { readKey, writeKey } from '@/lib/storage'
import { dispatchPluginNativeNotification, type PluginNativeNotificationInput } from '@/store/native-notifications'

import { createPluginViewerActions, type PluginViewerInput } from './plugin-viewer'
import { registry } from './registry'
export type { PluginViewerInput } from './plugin-viewer'
import type { Contribution } from './types'

export type { PluginRestOptions } from '@/hermes'
export type { HermesOpenTarget } from '@/lib/hermes-open-target'
export type { PluginNativeNotificationInput, PluginNotificationAction } from '@/store/native-notifications'

/** A contribution as a plugin author writes it — provenance + id scoping are
 *  the host's job, so those fields are off-limits here. */
export type PluginContribution = Omit<Contribution, 'source' | 'id'> & { id: string }

/** Namespaced JSON persistence (the VS Code `globalState` analog). Keys live
 *  under `hermes.plugin.<id>.` — plugins can't read or clobber each other. */
export interface PluginStorage {
  get<T>(key: string, fallback: T): T
  set(key: string, value: unknown): void
  remove(key: string): void
}

/** The curated OS door — every way a plugin reaches outside the app window,
 *  in one attributed namespace instead of the raw `window.hermesDesktop`
 *  bridge. Every member resolves a result instead of throwing when the
 *  capability can't apply (no Electron shell, older desktop build), so
 *  callers branch on the return value rather than sniffing the bridge. */
export interface PluginOs {
  /** Open/update an owned sandboxed native URL viewer without focusing it. */
  openViewer: (input: PluginViewerInput) => Promise<boolean>
  closeViewer: (id: string) => Promise<boolean>
  /** Native OS notification (Electron), attributed to this plugin. Gated by
   *  Settings ▸ Notifications ▸ "Plugin notifications" and fires only while
   *  the user is away from Hermes — use `host.notify` for the in-app toast.
   *  Throttled per plugin; reserve it for genuinely notable events.
   *  Supports `icon`, `activate` (e.g. `hermes://index-network/intent/1`),
   *  action buttons, and renderer `onActivate` / `onAction` callbacks. */
  notify: (input: PluginNativeNotificationInput) => void
  /** Open a URL with the OS default handler (browser, mail client, custom
   *  schemes like `spotify:`). Resolves false when the shell can't. */
  openExternal: (url: string) => Promise<boolean>
  /** Reveal a path in the OS file manager (Finder / Explorer). Resolves
   *  false when unavailable. */
  revealPath: (path: string) => Promise<boolean>
  /** Native save dialog. Resolves the chosen path, or null on cancel /
   *  when unavailable. The path is on the BACKEND's filesystem, so hand it
   *  to a `rest` call rather than trying to write it from the renderer. */
  pickSavePath: (options?: PluginFileDialogOptions) => Promise<null | string>
  /** Native open dialog, single file. Resolves the chosen path, or null on
   *  cancel / when unavailable. */
  pickOpenPath: (options?: PluginFileDialogOptions) => Promise<null | string>
  /** Write text to the system clipboard. Resolves false when unavailable. */
  writeClipboard: (text: string) => Promise<boolean>
}

export interface PluginFileDialogOptions {
  defaultPath?: string
  filters?: Array<{ extensions: string[]; name: string }>
  title?: string
}

export interface PluginContext {
  /** The resolved plugin source tag, e.g. `'plugin:cost-meter'`. */
  readonly source: string
  /** Register one contribution (id namespaced, source stamped). */
  register: (c: PluginContribution) => () => void
  /** Register several at once; the returned disposer removes all of them. */
  registerMany: (cs: PluginContribution[]) => () => void
  /** Register an arbitrary cleanup to run on unload/disable — for side effects
   *  that aren't contributions or sockets (store subscriptions, timers). Runs
   *  alongside every other disposer when the plugin deactivates. */
  onDispose: (fn: () => void) => void
  /** REST to this plugin's own backend namespace (`/api/plugins/<id>`); `path`
   *  is relative ('/board'). The sanctioned door for a plugin that ships a
   *  `plugin_api.py` — profile-aware, namespace-scoped by construction. Use
   *  `host.request` for gateway JSON-RPC. */
  rest: <T>(path: string, opts?: PluginRestOptions) => Promise<T>
  /** Live twin of `rest`: a WebSocket to this plugin's own namespace
   *  ('/events'), JSON frames to `onMessage`, auto-reconnect, disposer
   *  returned. Resolves to a no-op on OAuth remotes — treat it as an
   *  accelerator over your polling, never a replacement. */
  socket: (path: string, onMessage: (data: unknown) => void) => () => void
  /** The curated OS door: native notification, open-external, reveal-in-file-
   *  manager, clipboard — attributed to this plugin, result-shaped (never
   *  throws for a missing capability). */
  os: PluginOs
  /** Plugin-scoped persistence. */
  storage: PluginStorage
  /** Plugin-scoped i18n: ship + register locale bundles under this plugin,
   *  resolved against the app's active locale — no core `en.ts` edit. */
  i18n: PluginI18n
}

export interface HermesPlugin {
  /** Stable slug — becomes the `plugin:<id>` source and the id namespace. */
  id: string
  /** Human name for settings / about UI. */
  name?: string
  /** One-liner for the settings inventory (what the plugin adds). */
  description?: string
  /** Registers on load when the user hasn't chosen (default true). Set false
   *  for opt-in plugins: they inventory in Settings ▸ Plugins, off until the
   *  user flips the switch. */
  defaultEnabled?: boolean
  /** Called once at load; wire contributions through `ctx`. */
  register: (ctx: PluginContext) => void
}

function createPluginStorage(pluginId: string): PluginStorage {
  const scoped = (key: string) => `hermes.plugin.${pluginId}.${key}`

  return {
    get(key, fallback) {
      const raw = readKey(scoped(key))

      if (raw === null) {
        return fallback
      }

      try {
        return JSON.parse(raw)
      } catch {
        return fallback
      }
    },
    set: (key, value) => writeKey(scoped(key), JSON.stringify(value)),
    remove: key => writeKey(scoped(key), null)
  }
}

// Never throws for a missing capability: the renderer can outlive an older
// Electron shell (or run in a plain browser), so every door degrades to a
// false result the plugin can branch on.
function createPluginOs(pluginId: string, track: (dispose: () => void) => void): PluginOs {
  const attempt = async (run: (bridge: NonNullable<typeof window.hermesDesktop>) => Promise<boolean>) => {
    const bridge = typeof window === 'undefined' ? undefined : window.hermesDesktop

    if (!bridge) {
      return false
    }

    try {
      return await run(bridge)
    } catch {
      return false
    }
  }

  // Same shape as `attempt`, for the pickers that answer with a path.
  const attemptPath = async (run: (bridge: NonNullable<typeof window.hermesDesktop>) => Promise<null | string>) => {
    const bridge = typeof window === 'undefined' ? undefined : window.hermesDesktop

    if (!bridge) {
      return null
    }

    try {
      return await run(bridge)
    } catch {
      return null
    }
  }

  return {
    ...createPluginViewerActions(pluginId, track),
    notify: input => dispatchPluginNativeNotification(pluginId, input),
    openExternal: url =>
      attempt(async bridge => {
        await bridge.openExternal(url)

        return true
      }),
    pickOpenPath: options =>
      attemptPath(async bridge => {
        const picked = await bridge.selectPaths?.({ ...options, multiple: false })

        return picked?.[0] ?? null
      }),
    pickSavePath: options => attemptPath(async bridge => (await bridge.selectSavePath?.(options)) ?? null),
    revealPath: path => attempt(async bridge => (bridge.revealPath ? bridge.revealPath(path) : false)),
    writeClipboard: text => attempt(bridge => bridge.writeClipboard(text))
  }
}

/** Build the scoped context handed to a plugin's `register`. `onDispose`
 *  receives every registration's disposer (the loader's unload/reload hook). */
export function createPluginContext(pluginId: string, onDispose?: (dispose: () => void) => void): PluginContext {
  const source = `plugin:${pluginId}`
  const scope = (c: PluginContribution): Contribution => ({ ...c, id: `${pluginId}:${c.id}`, source })

  const track = (dispose: () => void) => {
    onDispose?.(dispose)

    return dispose
  }

  return {
    source,
    register: c => track(registry.register(scope(c))),
    registerMany: cs => track(registry.registerMany(cs.map(scope))),
    onDispose: fn => void track(fn),
    rest: <T>(path: string, opts?: PluginRestOptions) => pluginRest<T>(pluginId, path, opts),
    socket: (path, onMessage) => track(pluginSocket(pluginId, path, onMessage)),
    os: createPluginOs(pluginId, track),
    storage: createPluginStorage(pluginId),
    i18n: createPluginI18n(pluginId, track)
  }
}
