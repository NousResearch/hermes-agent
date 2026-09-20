import { atom } from 'nanostores'

import { captureOwner, type OwnerScope, type ProfileScope, profileScopeKey, scopedApi } from '@/api/client'
import { hermesConfigSchemaKey } from '@/app/hooks/use-config-record'
import { queryClient } from '@/lib/query-client'
import { requestGatewayForAgent } from '@/store/gateway'
import { notifyError } from '@/store/notifications'
import type { MemoryStatusResponse } from '@/types/hermes'

/**
 * Feature store for backend (agent) plugins — the native Hermes plugins plus
 * portable Agent Plugins v1 packages the backend discovers on disk. Settings
 * renders this next to the desktop (renderer) plugin inventory so every plugin
 * the user has is discoverable and toggleable from one page, whatever process
 * it runs in.
 *
 * Backed by the gateway's `plugins.manage` RPC — the same list/toggle
 * primitives `hermes plugins` and the dashboard use, so all surfaces agree on
 * what's installed and what's enabled. Works against every backend topology
 * (local spawn, SSH, URL+token) because it rides the session's own transport.
 */

export interface AgentPluginRow {
  name: string
  /** Canonical registry key (e.g. `image_gen/fal`) — absent on legacy backends. */
  key?: string
  version: string
  description: string
  /** 'bundled' | 'user' | 'git' | 'project' | 'entrypoint' */
  source: string
  status: 'enabled' | 'disabled' | 'not enabled'
  /** Agent Plugins v1 package (portable skills/MCP format) vs native Hermes. */
  portable?: boolean
  /** Curated-catalog provenance (from the install sidecar), when present. */
  catalog_name?: string
  catalog_tier?: string
  installed_sha?: string
  /** Current catalog pin for this entry (backend-computed). */
  catalog_sha?: string
  /** Human label the catalog attaches to that pin ("1.4.0"); shown on the Update button when present. */
  catalog_version?: string | null
  /** Installed SHA differs from the catalog pin — an update is available. */
  update_available?: boolean
  /** Full commit SHA a `--ref` install is pinned to (custom sources; refuses `update`). */
  pinned_sha?: string
  /** The package folder also ships `desktop/plugin.js` (unified agent+desktop package). */
  has_desktop_half?: boolean
  /** Absolute install dir on the backend (informational). */
  install_dir?: string
}

/** A `--ref` pin is a full 40-hex commit SHA; branches and tags are refused server-side. */
export const COMMIT_SHA_RE = /^[0-9a-f]{40}$/i

export type AgentPluginsStatus = 'idle' | 'loading' | 'ready' | 'error'

type GatewayRequest = <T>(method: string, params?: Record<string, unknown>) => Promise<T>

export const $agentPlugins = atom<AgentPluginRow[]>([])
export const $agentPluginsStatus = atom<AgentPluginsStatus>('idle')
export const $agentPluginsError = atom<string | null>(null)
/** Best available address of the row whose toggle RPC is in flight. */
export const $agentPluginBusy = atom<string | null>(null)

// Rows the Plugins page actually lists (and search should surface): plugins
// the USER installed. Repo-bundled built-ins ship enabled-by-default and are
// configured from their own surfaces, so they're pure noise here. The prefix
// list is the fallback for older backends whose rows predate a reliable
// `source` field — same curation stance as desktop-slash-commands.ts.
const HIDDEN_KEY_PREFIXES = ['dashboard_auth/', 'model-providers/', 'platforms/']

export const isDesktopRelevantPlugin = (row: AgentPluginRow): boolean => {
  if (row.source === 'bundled') {
    return false
  }

  const key = row.key

  return !key || !HIDDEN_KEY_PREFIXES.some(prefix => key.startsWith(prefix))
}

const inflightLoads = new Map<string, Promise<void>>()
let foregroundOwner: string | null = null

const requestForOwner =
  (owner: OwnerScope): GatewayRequest =>
  (method, params) =>
    requestGatewayForAgent(owner.connectionId, owner.profile, method, { ...params, profile: owner.profile })

// A foreground load re-homes the store to `profile`'s owner; a background load publishes only if that owner is still foreground.
export function loadAgentPlugins(profile?: ProfileScope, foreground = true): Promise<void> {
  const owner = captureOwner(profile)
  const key = profileScopeKey(owner)

  const canPublish = () => foregroundOwner === key

  if (foreground && !canPublish()) {
    // Installed-state guards read the rows while loading, so the previous owner's rows are cleared before the read starts.
    foregroundOwner = key
    $agentPlugins.set([])
    $agentPluginsError.set(null)
    $agentPluginsStatus.set('loading')
  } else if (canPublish() && $agentPluginsStatus.get() !== 'ready') {
    $agentPluginsStatus.set('loading')
  }

  const existing = inflightLoads.get(key)

  if (existing) {
    return existing
  }

  const flight = (async () => {
    try {
      const result = await requestForOwner(owner)<{ plugins?: AgentPluginRow[] }>('plugins.manage', { action: 'list' })

      if (canPublish()) {
        $agentPlugins.set(result?.plugins ?? [])
        $agentPluginsStatus.set('ready')
        $agentPluginsError.set(null)
      }
    } catch (e) {
      if (canPublish()) {
        $agentPluginsError.set(e instanceof Error ? e.message : String(e))
        $agentPluginsStatus.set('error')
      }
    } finally {
      inflightLoads.delete(key)
    }
  })()

  inflightLoads.set(key, flight)

  return flight
}

// A mutation needs a scan started after its write, not one already in flight; the owner's memory and schema reads go stale.
async function refreshAfterMutation(owner: OwnerScope): Promise<void> {
  const key = profileScopeKey(owner)
  const schemaKey = hermesConfigSchemaKey(owner)
  await inflightLoads.get(key)
  await loadAgentPlugins(owner, false)
  void queryClient.invalidateQueries({
    refetchType: 'none',
    predicate: ({ queryKey }) =>
      queryKey[0] === 'memory-status' || queryKey[0] === 'memory-provider-config'
        ? queryKey[1] === key
        : queryKey[0] === 'hermes-config-schema' &&
          (queryKey.length === 1 || JSON.stringify(queryKey) === JSON.stringify(schemaKey))
  })
}

// Whether the backend now lists `providerId` as installed code; a status read started before the install must not overwrite this one.
export async function discoverInstalledMemoryProvider(owner: OwnerScope, providerId: string): Promise<boolean> {
  const queryKey = ['memory-status', profileScopeKey(owner)]
  await queryClient.cancelQueries({ queryKey, exact: true })
  const status = await scopedApi<MemoryStatusResponse>(owner, { path: '/api/memory' })
  queryClient.setQueryData(queryKey, status)

  return status.providers.some(provider => provider.name === providerId && provider.status !== 'missing')
}

/** Flip a backend plugin on/off and patch the row from the RPC's refreshed
 *  copy. Addressed by canonical key ONLY — bare names collide across category
 *  dirs (image_gen/fal vs video_gen/fal), which is exactly why the backend
 *  moved to key-addressed toggles. Rows without a key (pre-contract-v6
 *  backends) render read-only instead of falling back to the collision-prone
 *  name protocol; the backend-contract skew toast points the user at the
 *  update. Returns whether the toggle stuck. */
export async function toggleAgentPlugin(
  key: string,
  enable: boolean,
  failMessage: string,
  profile?: ProfileScope
): Promise<boolean> {
  const owner = captureOwner(profile)
  $agentPluginBusy.set(key)

  try {
    const result = await requestForOwner(owner)<{ ok?: boolean }>('plugins.manage', { action: 'toggle', key, enable })

    if (!result?.ok) {
      throw new Error(failMessage)
    }

    await refreshAfterMutation(owner)

    return true
  } catch (e) {
    notifyError(e, failMessage)

    return false
  } finally {
    $agentPluginBusy.set(null)
  }
}

export interface AgentPluginInstallResult {
  ok: boolean
  pluginName?: string
  warnings?: string[]
  missingEnv?: string[]
  error?: string
}

export async function installAgentPlugin(opts: {
  identifier: string
  force?: boolean
  enable?: boolean
  /** Curated-catalog install: the backend resolves repo + pinned SHA from
   *  its own plugin-catalog and records provenance in the sidecar. */
  catalogName?: string
  /** Pin a custom source to one full commit SHA (team-wide reproducible install). */
  ref?: string
  /** Target profile's HERMES_HOME (null/undefined = backend launch profile). */
  profile?: ProfileScope
}): Promise<AgentPluginInstallResult> {
  const owner = captureOwner(opts.profile)

  try {
    const result = await requestForOwner(owner)<{
      ok?: boolean
      plugin_name?: string
      warnings?: string[]
      missing_env?: string[]
      error?: string
    }>('plugins.manage', {
      action: 'install',
      identifier: opts.identifier,
      force: Boolean(opts.force),
      enable: opts.enable ?? true,
      ...(opts.catalogName ? { catalog_name: opts.catalogName } : {}),
      ...(opts.ref ? { ref: opts.ref } : {})
    })

    if (!result?.ok) {
      return { ok: false, error: result?.error || 'Install failed' }
    }

    await refreshAfterMutation(owner)

    return {
      ok: true,
      pluginName: result.plugin_name,
      warnings: result.warnings,
      missingEnv: result.missing_env
    }
  } catch (e) {
    return { ok: false, error: e instanceof Error ? e.message : String(e) }
  }
}

/** Re-pin a catalog-installed plugin to the current catalog SHA (backend
 *  `plugins.manage update`; catalog installs only). Refreshes the list on
 *  success. Returns whether the update applied. */
export async function updateAgentPlugin(name: string, failMessage: string, profile?: ProfileScope): Promise<boolean> {
  const owner = captureOwner(profile)
  $agentPluginBusy.set(name)

  try {
    const result = await requestForOwner(owner)<{ ok?: boolean; unchanged?: boolean }>('plugins.manage', {
      action: 'update',
      name
    })

    if (!result?.ok) {
      throw new Error(failMessage)
    }

    await refreshAfterMutation(owner)

    return !result.unchanged
  } catch (e) {
    notifyError(e, failMessage)

    return false
  } finally {
    $agentPluginBusy.set(null)
  }
}
