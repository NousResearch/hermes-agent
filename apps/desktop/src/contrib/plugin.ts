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
import { readKey, writeKey } from '@/lib/storage'

import { registry } from './registry'
import type { Contribution } from './types'

export type { PluginRestOptions } from '@/hermes'

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

export interface PluginContext {
  /** The resolved plugin source tag, e.g. `'plugin:cost-meter'`. */
  readonly source: string
  /** Register one contribution (id namespaced, source stamped). */
  register: (c: PluginContribution) => () => void
  /** Register several at once; the returned disposer removes all of them. */
  registerMany: (cs: PluginContribution[]) => () => void
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
  /** Plugin-scoped persistence. */
  storage: PluginStorage
}

export interface HermesPlugin {
  /** Stable slug — becomes the `plugin:<id>` source and the id namespace. */
  id: string
  /** Human name for settings / about UI. */
  name?: string
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
    rest: <T>(path: string, opts?: PluginRestOptions) => pluginRest<T>(pluginId, path, opts),
    socket: (path, onMessage) => track(pluginSocket(pluginId, path, onMessage)),
    storage: createPluginStorage(pluginId)
  }
}
