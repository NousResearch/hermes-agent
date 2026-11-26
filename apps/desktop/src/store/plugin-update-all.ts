import type { PluginsManageResult, PluginUpdateAllRow } from '@hermes/shared'
import { atom } from 'nanostores'

import { type GatewayRequest, loadAgentPlugins } from '@/store/agent-plugins'

/**
 * "Update all" and the auto-update switch of the Plugins tab, over `plugins.manage update_all` /
 * `auto_update`: the same unattended path as `/plugins update` and the dashboard. Nothing is
 * accepted on the user's behalf; a widened pin or caution scan comes back `needs_review` and the
 * plugin's own Update button walks that review.
 */

export interface PluginAutoUpdateState {
  autoApply: boolean
  checkHours: number
  /** The profile this state was read for; a scope switch must not show another profile's switch. */
  profile: null | string
}

export interface PluginUpdateAllSummary {
  failed: number
  review: number
  rows: PluginUpdateAllRow[]
  updated: number
}

/** Profiles whose update-all is running (the default profile is `''`): a run for one profile leaves
 *  another profile's button enabled. */
export const $pluginUpdateAllRunning = atom<ReadonlySet<string>>(new Set())

const profileKey = (profile: null | string) => profile ?? ''

export const isPluginUpdateAllRunning = (running: ReadonlySet<string>, profile: null | string) =>
  running.has(profileKey(profile))

function markRunning(profile: null | string, on: boolean) {
  const next = new Set($pluginUpdateAllRunning.get())

  if (on) {
    next.add(profileKey(profile))
  } else {
    next.delete(profileKey(profile))
  }

  $pluginUpdateAllRunning.set(next)
}

export const $pluginAutoUpdate = atom<PluginAutoUpdateState | null>(null)

// One minute per plugin is the single-update budget; a whole sweep gets ten.
const UPDATE_ALL_TIMEOUT_MS = 600_000

const scoped = (params: Record<string, unknown>, profile: null | string) => (profile ? { ...params, profile } : params)

const countOf = (rows: PluginUpdateAllRow[], status: PluginUpdateAllRow['status']) =>
  rows.filter(row => row.status === status).length

export async function updateAllAgentPlugins(
  request: GatewayRequest,
  profile: null | string
): Promise<PluginUpdateAllSummary> {
  markRunning(profile, true)

  try {
    const result = await request<PluginsManageResult>(
      'plugins.manage',
      scoped({ action: 'update_all' }, profile),
      UPDATE_ALL_TIMEOUT_MS
    )

    const rows = result?.results ?? []

    if (countOf(rows, 'updated')) {
      await loadAgentPlugins(request, profile)
    }

    return {
      rows,
      updated: countOf(rows, 'updated'),
      review: countOf(rows, 'needs_review'),
      failed: countOf(rows, 'failed')
    }
  } finally {
    markRunning(profile, false)
  }
}

const toState = (result: PluginsManageResult | undefined, profile: null | string): PluginAutoUpdateState => ({
  autoApply: Boolean(result?.auto_apply),
  checkHours: result?.check_hours ?? 24,
  profile
})

export async function loadPluginAutoUpdate(request: GatewayRequest, profile: null | string): Promise<void> {
  const result = await request<PluginsManageResult>('plugins.manage', scoped({ action: 'auto_update' }, profile))
  $pluginAutoUpdate.set(toState(result, profile))
}

export async function setPluginAutoUpdate(
  request: GatewayRequest,
  profile: null | string,
  enable: boolean
): Promise<void> {
  const result = await request<PluginsManageResult>(
    'plugins.manage',
    scoped({ action: 'auto_update', enable }, profile)
  )

  $pluginAutoUpdate.set(toState(result, profile))
}
