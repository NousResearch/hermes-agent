import { atom } from 'nanostores'

import { notifyError } from '@/store/notifications'

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
}

export type AgentPluginsStatus = 'idle' | 'loading' | 'ready' | 'error'

/** The recovering `requestGateway` from `useGatewayRequest`. */
export type GatewayRequest = <T>(method: string, params?: Record<string, unknown>) => Promise<T>

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

let inflight: Promise<void> | null = null
let inflightProfile: string | null = null
// Bumped per load so a slow response from a previous profile scope can't
// overwrite the newer scope's list (async results can land out of order).
let loadGeneration = 0

/** Scope a `plugins.manage` payload to a profile. Omitted (null) = the
 *  backend's launch profile — older backends ignore the extra param. */
const withProfile = (params: Record<string, unknown>, profile?: string | null) =>
  profile ? { ...params, profile } : params

/** Fetch the backend plugin list, optionally scoped to another profile's
 *  HERMES_HOME. Always refetches (it's a cheap local disk scan on the
 *  backend); concurrent callers for the SAME profile share one in-flight
 *  request — a different profile starts fresh so a scope switch can't get a
 *  stale list. */
export function loadAgentPlugins(request: GatewayRequest, profile?: string | null): Promise<void> {
  const scope = profile ?? null

  if (inflight && inflightProfile === scope) {
    return inflight
  }

  const generation = ++loadGeneration

  inflightProfile = scope
  inflight = (async () => {
    if ($agentPluginsStatus.get() !== 'ready') {
      $agentPluginsStatus.set('loading')
    }

    try {
      const result = await request<{ plugins?: AgentPluginRow[] }>(
        'plugins.manage',
        withProfile({ action: 'list' }, scope)
      )

      if (generation !== loadGeneration) {
        return
      }

      $agentPlugins.set(result?.plugins ?? [])
      $agentPluginsStatus.set('ready')
      $agentPluginsError.set(null)
    } catch (e) {
      if (generation !== loadGeneration) {
        return
      }

      $agentPluginsError.set(e instanceof Error ? e.message : String(e))
      $agentPluginsStatus.set('error')
    } finally {
      if (generation === loadGeneration) {
        inflight = null
        inflightProfile = null
      }
    }
  })()

  return inflight
}

/** Flip a backend plugin on/off and patch the row from the RPC's refreshed
 *  copy. Addressed by canonical key ONLY — bare names collide across category
 *  dirs (image_gen/fal vs video_gen/fal), which is exactly why the backend
 *  moved to key-addressed toggles. Rows without a key (pre-contract-v6
 *  backends) render read-only instead of falling back to the collision-prone
 *  name protocol; the backend-contract skew toast points the user at the
 *  update. Returns whether the toggle stuck. */
export async function toggleAgentPlugin(
  request: GatewayRequest,
  key: string,
  enable: boolean,
  failMessage: string,
  profile?: string | null
): Promise<boolean> {
  $agentPluginBusy.set(key)

  try {
    const result = await request<{ ok?: boolean; plugin?: AgentPluginRow | null }>(
      'plugins.manage',
      withProfile(
        {
          action: 'toggle',
          key,
          enable
        },
        profile
      )
    )

    if (!result?.ok) {
      throw new Error(failMessage)
    }

    const refreshed = result.plugin

    if (refreshed) {
      $agentPlugins.set($agentPlugins.get().map(row => (row.key === key ? { ...row, ...refreshed } : row)))
    } else {
      await loadAgentPlugins(request, profile)
    }

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

export async function installAgentPlugin(
  request: GatewayRequest,
  opts: { identifier: string; force?: boolean; enable?: boolean }
): Promise<AgentPluginInstallResult> {
  try {
    const result = await request<{
      ok?: boolean
      plugin_name?: string
      warnings?: string[]
      missing_env?: string[]
      error?: string
    }>('plugins.manage', {
      action: 'install',
      identifier: opts.identifier,
      force: Boolean(opts.force),
      enable: opts.enable ?? true
    })

    if (!result?.ok) {
      return { ok: false, error: result?.error || 'Install failed' }
    }

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
