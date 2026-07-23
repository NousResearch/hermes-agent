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
import { clearPetSignal, clearPetSignals, type PetSignalState, upsertPetSignal } from '@/store/pet-signals'
import { $activeGatewayProfile, normalizeProfileKey } from '@/store/profile'

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

export const PLUGIN_PET_ACTIVITY_LIMITS = {
  maxIdsPerSource: 64,
  maxIdLength: 128,
  maxPriority: 100,
  maxTtlMs: 300_000,
  minPriority: 0,
  minTtlMs: 1_000
} as const

export type PluginPetActivityState = PetSignalState

/** A deliberately narrow, host-owned animation request. Plugins cannot spoof
 * another source, create persistent activity, or inject copy/actions into the
 * mascot surface. Native Hermes activity remains authoritative. */
export interface PluginPetActivityInput {
  readonly id: string
  readonly priority?: number
  readonly state: PluginPetActivityState
  readonly ttlMs: number
}

export interface PluginPetSignals {
  /** Publish or replace one bounded activity identity. Returns a guarded clear. */
  publishActivity: (input: PluginPetActivityInput) => () => void
  /** Clear one identity owned by this plugin. */
  clearActivity: (id: string) => void
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
  /** Plugin-scoped i18n: ship + register locale bundles under this plugin,
   *  resolved against the app's active locale — no core `en.ts` edit. */
  i18n: PluginI18n
  /** Bounded, expiring mascot activity owned by this plugin. */
  pet: PluginPetSignals
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

const PLUGIN_PET_ACTIVITY_FIELDS = new Set(['id', 'priority', 'state', 'ttlMs'])
const PLUGIN_PET_ACTIVITY_STATES = new Set<PluginPetActivityState>(['blocked', 'done', 'failed', 'thinking', 'working'])

// Process-lifetime source generations and identity sets deliberately survive a
// plugin hot reload. That keeps new publications ahead of retained tombstones
// and prevents reloads from bypassing the per-source identity bound.
const pluginPetGenerations = new Map<string, number>()
const pluginPetIdentities = new Map<string, Set<string>>()

function validatePluginPetActivityId(id: unknown): string {
  if (
    typeof id !== 'string' ||
    id.length === 0 ||
    id.length > PLUGIN_PET_ACTIVITY_LIMITS.maxIdLength ||
    id.trim() !== id
  ) {
    throw new Error('Plugin pet activity id must be a non-empty, trimmed, bounded string')
  }

  return id
}

function validatePluginPetActivity(input: PluginPetActivityInput): Required<PluginPetActivityInput> {
  if (!input || typeof input !== 'object') {
    throw new Error('Plugin pet activity must be an object')
  }

  const unsupported = Object.keys(input).find(field => !PLUGIN_PET_ACTIVITY_FIELDS.has(field))

  if (unsupported) {
    throw new Error(`Unsupported field in plugin pet activity: ${unsupported}`)
  }

  const id = validatePluginPetActivityId(input.id)

  if (!PLUGIN_PET_ACTIVITY_STATES.has(input.state)) {
    throw new Error('Unsupported plugin pet activity state')
  }

  if (
    !Number.isInteger(input.ttlMs) ||
    input.ttlMs < PLUGIN_PET_ACTIVITY_LIMITS.minTtlMs ||
    input.ttlMs > PLUGIN_PET_ACTIVITY_LIMITS.maxTtlMs
  ) {
    throw new Error('Plugin pet activity TTL is outside the supported range')
  }

  const priority = input.priority ?? PLUGIN_PET_ACTIVITY_LIMITS.minPriority

  if (
    !Number.isInteger(priority) ||
    priority < PLUGIN_PET_ACTIVITY_LIMITS.minPriority ||
    priority > PLUGIN_PET_ACTIVITY_LIMITS.maxPriority
  ) {
    throw new Error('Plugin pet activity priority is outside the supported range')
  }

  return { id, priority, state: input.state, ttlMs: input.ttlMs }
}

function createPluginPetSignals(source: string, track: (dispose: () => void) => () => void): PluginPetSignals {
  let active = true
  let profile = normalizeProfileKey($activeGatewayProfile.get())

  const stopProfile = $activeGatewayProfile.subscribe(next => {
    const nextProfile = normalizeProfileKey(next)

    if (nextProfile !== profile) {
      profile = nextProfile
      clearPetSignals(source)
    }
  })

  track(() => {
    if (!active) {
      return
    }

    active = false
    stopProfile()
    clearPetSignals(source)
  })

  const assertActive = () => {
    if (!active) {
      throw new Error('Plugin pet activity publisher is disposed')
    }
  }

  return {
    clearActivity(id) {
      assertActive()
      clearPetSignal(source, validatePluginPetActivityId(id))
    },
    publishActivity(input) {
      assertActive()
      const activity = validatePluginPetActivity(input)
      let identities = pluginPetIdentities.get(source)

      if (!identities) {
        identities = new Set()
        pluginPetIdentities.set(source, identities)
      }

      if (!identities.has(activity.id)) {
        if (identities.size >= PLUGIN_PET_ACTIVITY_LIMITS.maxIdsPerSource) {
          throw new Error('Plugin pet activity identity limit reached')
        }

        identities.add(activity.id)
      }

      const now = Date.now()
      const createdAt = Math.max(now, (pluginPetGenerations.get(source) ?? Number.NEGATIVE_INFINITY) + 1)
      pluginPetGenerations.set(source, createdAt)
      upsertPetSignal({
        createdAt,
        expiresAt: now + activity.ttlMs,
        id: activity.id,
        priority: activity.priority,
        source,
        state: activity.state
      })

      let cleared = false

      return () => {
        if (cleared) {
          return
        }

        cleared = true
        clearPetSignal(source, activity.id, createdAt)
      }
    }
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

  const pet = createPluginPetSignals(source, track)

  return {
    source,
    register: c => track(registry.register(scope(c))),
    registerMany: cs => track(registry.registerMany(cs.map(scope))),
    rest: <T>(path: string, opts?: PluginRestOptions) => pluginRest<T>(pluginId, path, opts),
    socket: (path, onMessage) => track(pluginSocket(pluginId, path, onMessage)),
    storage: createPluginStorage(pluginId),
    i18n: createPluginI18n(pluginId, track),
    pet
  }
}
