/**
 * Achievements data layer. Everything goes through `ctx.rest` — the plugin's
 * own `/api/plugins/hermes-achievements/*` FastAPI router
 * (`plugins/hermes-achievements/dashboard/plugin_api.py`), reused as-is via
 * the desktop's namespace-scoped REST door. No new backend.
 *
 * Fetching, caching, polling, dedupe, and invalidation are React Query's job
 * (the app's standard, via the SDK). This module owns the query keys and the
 * REST calls.
 */

import { type PluginRestOptions, queryClient } from '@hermes/plugin-sdk'

import type { AchievementFilter, AchievementsResponse } from './types'

type Rest = <T>(path: string, opts?: PluginRestOptions) => Promise<T>

let rest: null | Rest = null

export const achievementsKey = (filter?: AchievementFilter) => ['hermes-achievements', filter ?? 'all'] as const

/** Bind the plugin's REST door at register time and return a disposer the host
 *  runs on unload/disable — so no stale reference survives a toggle. */
export function bindApi(r: Rest): () => void {
  rest = r

  return () => {
    rest = null
  }
}

/** Fetch the full evaluated achievements payload. */
export function fetchAchievements(): Promise<AchievementsResponse> {
  if (!rest) {
    return Promise.reject(new Error('achievements rest door not bound'))
  }

  return rest<AchievementsResponse>('/achievements')
}

/** Trigger a forced backend rescan, then invalidate the shared cache. */
export async function rescanAchievements(): Promise<void> {
  if (!rest) {
    throw new Error('achievements rest door not bound')
  }

  await rest<{ ok: boolean }>('/rescan', { method: 'POST' })
  await queryClient.invalidateQueries({ queryKey: ['hermes-achievements'] })
}
