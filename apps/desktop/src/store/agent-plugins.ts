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

let inflight: Promise<void> | null = null

/** Fetch the backend plugin list. Always refetches (it's a cheap local disk
 *  scan on the backend); concurrent callers share one in-flight request. */
export function loadAgentPlugins(request: GatewayRequest): Promise<void> {
  if (inflight) {
    return inflight
  }

  inflight = (async () => {
    if ($agentPluginsStatus.get() !== 'ready') {
      $agentPluginsStatus.set('loading')
    }

    try {
      const result = await request<{ plugins?: AgentPluginRow[] }>('plugins.manage', { action: 'list' })
      $agentPlugins.set(result?.plugins ?? [])
      $agentPluginsStatus.set('ready')
      $agentPluginsError.set(null)
    } catch (e) {
      $agentPluginsError.set(e instanceof Error ? e.message : String(e))
      $agentPluginsStatus.set('error')
    } finally {
      inflight = null
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
  failMessage: string
): Promise<boolean> {
  $agentPluginBusy.set(key)

  try {
    const result = await request<{ ok?: boolean; plugin?: AgentPluginRow | null }>('plugins.manage', {
      action: 'toggle',
      key,
      enable
    })

    if (!result?.ok) {
      throw new Error(failMessage)
    }

    const refreshed = result.plugin

    if (refreshed) {
      $agentPlugins.set($agentPlugins.get().map(row => (row.key === key ? { ...row, ...refreshed } : row)))
    } else {
      await loadAgentPlugins(request)
    }

    return true
  } catch (e) {
    notifyError(e, failMessage)

    return false
  } finally {
    $agentPluginBusy.set(null)
  }
}
