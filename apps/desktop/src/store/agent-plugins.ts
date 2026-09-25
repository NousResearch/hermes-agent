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

export type AgentPluginServerState =
  | 'connected'
  | 'app_not_running'
  | 'endpoint_unavailable'
  | 'no_interactive_session'
  | 'version_too_old'
  | 'missing_app'
  | 'unknown'

export interface AgentPluginServer {
  name: string
  state: AgentPluginServerState
  sentence: string
}

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
  /** Manifest `config_schema` rendered as a settings form (with current values). */
  settings_schema?: PluginSettingField[]
  /** Full snapshot of declared application-backed MCP servers. */
  servers?: AgentPluginServer[]
}

export type PluginSettingFieldType = 'boolean' | 'enum' | 'json' | 'number' | 'secret' | 'string'

/** One `config_schema` key of a plugin manifest. Secrets never carry a value:
 *  `env` names the `.env` variable, `has_value` whether it is set. */
export interface PluginSettingField {
  key: string
  type: PluginSettingFieldType
  label: string
  description: string
  required: boolean
  value?: unknown
  default?: unknown
  choices?: string[]
  env?: string
  has_value?: boolean
}

export const normalizeAgentPluginRow = (row: AgentPluginRow): AgentPluginRow => ({
  ...row,
  servers: row.servers ?? []
})

/** A `--ref` pin is a full 40-hex commit SHA; branches and tags are refused server-side. */
export const COMMIT_SHA_RE = /^[0-9a-f]{40}$/i

export type AgentPluginsStatus = 'idle' | 'loading' | 'ready' | 'error'

/** The recovering `requestGateway` from `useGatewayRequest`. */
export type GatewayRequest = <T>(method: string, params?: Record<string, unknown>, timeoutMs?: number) => Promise<T>

export const $agentPlugins = atom<AgentPluginRow[]>([])
export const $agentPluginsStatus = atom<AgentPluginsStatus>('idle')
export const $agentPluginsError = atom<string | null>(null)
/** Best available address of the row whose toggle RPC is in flight. */
export const $agentPluginBusy = atom<string | null>(null)

// Rows the Plugins page actually lists (and search should surface): plugins
// the USER installed, plus the few repo-bundled lifecycle plugins that use the
// ordinary enable/disable contract and have no settings surface of their own
// (#98861). Every other built-in (providers, platforms, browser/web backends,
// dashboard auth, observability, unknown future keys) ships enabled-by-default
// and is configured from its own surface, so it's pure noise here. The prefix
// list is the fallback for older backends whose rows predate a reliable
// `source` field — same curation stance as desktop-slash-commands.ts.
const HIDDEN_KEY_PREFIXES = ['dashboard_auth/', 'model-providers/', 'platforms/']
const MANAGEABLE_BUNDLED_KEYS = new Set(['disk-cleanup', 'security-guidance'])

export const isDesktopRelevantPlugin = (row: AgentPluginRow): boolean => {
  if (row.source === 'bundled') {
    return MANAGEABLE_BUNDLED_KEYS.has(row.key ?? row.name)
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

      $agentPlugins.set((result?.plugins ?? []).map(normalizeAgentPluginRow))
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
      const snapshot = normalizeAgentPluginRow(refreshed)

      $agentPlugins.set($agentPlugins.get().map(row => (row.key === key ? { ...row, ...snapshot } : row)))
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
  /** The client stopped waiting; the backend may still finish the install. */
  timedOut?: boolean
  pluginName?: string
  warnings?: string[]
  missingEnv?: string[]
  error?: string
  /** What became usable in open chats of the profile (`activation.live_now`). */
  live: AgentPluginLiveNow
  /** Python tools or prompt sections that wait for the next chat (`activation.deferred`). */
  nextChat: boolean
}

export interface AgentPluginLiveServer {
  name: string
  connected: boolean
  tools: string[]
  error?: string | null
}

export interface AgentPluginLiveNow {
  mcpServers: AgentPluginLiveServer[]
  skills: string[]
}

const NO_LIVE: AgentPluginLiveNow = { mcpServers: [], skills: [] }

// Installing a catalog package can clone a repository and resolve Python dependencies.
// The ordinary Desktop RPC deadline is 30s, which can expire after the backend
// has already begun an install that will succeed. Keep this wait bounded while
// giving normal installs time to return their authoritative result.
const PLUGIN_INSTALL_REQUEST_TIMEOUT_MS = 120_000

export async function installAgentPlugin(
  request: GatewayRequest,
  opts: {
    identifier: string
    force?: boolean
    enable?: boolean
    /** Curated-catalog install: the backend resolves repo + pinned SHA from
     *  its own plugin-catalog and records provenance in the sidecar. */
    catalogName?: string
    /** Pin a custom source to one full commit SHA (team-wide reproducible install). */
    ref?: string
    /** Target profile's HERMES_HOME (null/undefined = backend launch profile). */
    profile?: string | null
  }
): Promise<AgentPluginInstallResult> {
  try {
    const result = await request<{
      ok?: boolean
      plugin_name?: string
      warnings?: string[]
      missing_env?: string[]
      activation?: {
        live_now?: {
          mcp_servers?: AgentPluginLiveServer[]
          skills?: { name: string }[]
        } | null
        deferred?: Record<string, string[]>
      } | null
      error?: string
    }>(
      'plugins.manage',
      withProfile(
        {
          action: 'install',
          identifier: opts.identifier,
          force: Boolean(opts.force),
          enable: opts.enable ?? true,
          ...(opts.catalogName ? { catalog_name: opts.catalogName } : {}),
          ...(opts.ref ? { ref: opts.ref } : {})
        },
        opts.profile
      ),
      PLUGIN_INSTALL_REQUEST_TIMEOUT_MS
    )

    if (!result?.ok) {
      return { ok: false, error: result?.error || 'Install failed', live: NO_LIVE, nextChat: false }
    }

    return {
      ok: true,
      pluginName: result.plugin_name,
      warnings: result.warnings,
      missingEnv: result.missing_env,
      live: {
        mcpServers: result.activation?.live_now?.mcp_servers ?? [],
        // `<namespace>:<skill>` is what the model loads; the toast shows the skill's own name.
        skills: (result.activation?.live_now?.skills ?? []).map(skill => skill.name.split(':').pop() ?? skill.name)
      },
      nextChat: Object.keys(result.activation?.deferred ?? {}).length > 0
    }
  } catch (e) {
    const message = e instanceof Error ? e.message : String(e)

    return {
      ok: false,
      timedOut: /^request timed out after \d+s: plugins\.manage$/.test(message),
      error: message,
      live: NO_LIVE,
      nextChat: false
    }
  }
}

/** Outcome of `updateAgentPlugin`: `applied` when the re-pin landed, `unchanged`
 *  when already at pin, `consent` when the new pin widens the plugin — the
 *  backend changed nothing and waits for `acceptCapabilities`. */
export type AgentPluginUpdateOutcome =
  { kind: 'applied' | 'unchanged' | 'failed' } | { kind: 'consent'; sha: string; deltaLines: string[] }

/** Re-pin a catalog-installed plugin to the current catalog SHA (backend
 *  `plugins.manage update`; catalog installs only). Refreshes the list on
 *  success. A pin that adds tools / hooks / deps / capabilities / a Desktop half
 *  comes back as `consent` with the delta; the caller confirms and retries with
 *  `acceptCapabilities`. */
export async function updateAgentPlugin(
  request: GatewayRequest,
  name: string,
  failMessage: string,
  profile?: string | null,
  acceptCapabilities = false
): Promise<AgentPluginUpdateOutcome> {
  $agentPluginBusy.set(name)

  try {
    const result = await request<{
      ok?: boolean
      unchanged?: boolean
      consent_required?: boolean
      sha?: string
      delta_lines?: string[]
    }>(
      'plugins.manage',
      withProfile({ action: 'update', name, ...(acceptCapabilities ? { accept_capabilities: true } : {}) }, profile)
    )

    if (result?.consent_required) {
      return { kind: 'consent', sha: (result.sha ?? '').slice(0, 8), deltaLines: result.delta_lines ?? [] }
    }

    if (!result?.ok) {
      throw new Error(failMessage)
    }

    await loadAgentPlugins(request, profile)

    return { kind: result.unchanged ? 'unchanged' : 'applied' }
  } catch (e) {
    notifyError(e, failMessage)

    return { kind: 'failed' }
  } finally {
    $agentPluginBusy.set(null)
  }
}

/** Uninstall a user-installed agent plugin (backend `plugins.manage remove`;
 *  deletes `<HERMES_HOME>/plugins/<name>` and its install metadata). Drops the
 *  row locally on success — callers rescan so a unified package's desktop half
 *  is pruned too. Returns whether the plugin was removed. */
export async function removeAgentPlugin(
  request: GatewayRequest,
  name: string,
  failMessage: string,
  profile?: string | null
): Promise<boolean> {
  $agentPluginBusy.set(name)

  try {
    const result = await request<{ ok?: boolean }>('plugins.manage', withProfile({ action: 'remove', name }, profile))

    if (!result?.ok) {
      throw new Error(failMessage)
    }

    $agentPlugins.set($agentPlugins.get().filter(row => row.name !== name))

    return true
  } catch (e) {
    notifyError(e, failMessage)

    return false
  } finally {
    $agentPluginBusy.set(null)
  }
}

export interface SaveAgentPluginSettingsOptions {
  /** Canonical plugin key (`plugins.entries.<key>`). */
  key: string
  /** Non-secret `config_schema` values, already coerced to their wire types. */
  values: Record<string, unknown>
  /** Secret fields: `.env` variable name → new value (blank = unchanged, never sent). */
  secrets: Record<string, string>
  /** Writes ONE secret through the credential route (`PUT /api/env`, profile-scoped
   *  by the caller) — secrets never ride the config.yaml RPC. */
  writeSecret: (env: string, value: string) => Promise<unknown>
  failMessage: string
  profile?: string | null
}

/** Persist a plugin's manifest-declared settings: values through
 *  `plugins.manage settings` (the backend writes `plugins.entries.<key>.settings`
 *  with the same writer `ctx.set_config` uses), secrets through `writeSecret`.
 *  Patches the row from the RPC's refreshed copy so the form re-reads what
 *  landed. Returns whether everything saved. */
export async function saveAgentPluginSettings(
  request: GatewayRequest,
  opts: SaveAgentPluginSettingsOptions
): Promise<boolean> {
  $agentPluginBusy.set(opts.key)

  try {
    for (const [env, value] of Object.entries(opts.secrets)) {
      if (value) {
        await opts.writeSecret(env, value)
      }
    }

    const result =
      Object.keys(opts.values).length > 0
        ? await request<{ ok?: boolean; plugin?: AgentPluginRow | null }>(
            'plugins.manage',
            withProfile({ action: 'settings', key: opts.key, values: opts.values }, opts.profile)
          )
        : null

    if (result && !result.ok) {
      throw new Error(opts.failMessage)
    }

    if (result?.plugin) {
      const refreshed = result.plugin

      $agentPlugins.set($agentPlugins.get().map(row => (row.key === opts.key ? { ...row, ...refreshed } : row)))
    } else {
      await loadAgentPlugins(request, opts.profile)
    }

    return true
  } catch (e) {
    notifyError(e, opts.failMessage)

    return false
  } finally {
    $agentPluginBusy.set(null)
  }
}
