/**
 * Achievements data boundary. The Desktop plugin delegates all catalog,
 * evaluation, persistence, and scanning work to the canonical dashboard
 * backend through the namespace-scoped `ctx.rest` door.
 */

import type { PluginRestOptions } from '@hermes/plugin-sdk'

import type { AchievementsResponse, RescanResponse, ScanMeta } from './types'

type Rest = <T>(path: string, opts?: PluginRestOptions) => Promise<T>

let rest: null | Rest = null

export const ACHIEVEMENTS_KEY = ['hermes-achievements', 'achievements'] as const
export const ACTIVE_SCAN_POLL_MS = 1_500

export function bindAchievementsApi(boundRest: Rest): () => void {
  rest = boundRest

  return () => {
    if (rest === boundRest) {
      rest = null
    }
  }
}

function call<T>(path: string, opts?: PluginRestOptions): Promise<T> {
  return rest ? rest<T>(path, opts) : Promise.reject(new Error('achievements api not ready'))
}

export function fetchAchievements(): Promise<AchievementsResponse> {
  return call<AchievementsResponse>('/achievements')
}

export async function rescanAchievements(): Promise<RescanResponse> {
  const result = await call<Omit<RescanResponse, 'is_stale'>>('/rescan', { method: 'POST' })

  return { ...result, is_stale: false }
}

export function scanIsActive(scanMeta?: ScanMeta): boolean {
  const mode = scanMeta?.mode?.toLowerCase()
  const state = scanMeta?.status?.state?.toLowerCase()

  return mode === 'pending' || mode === 'in_progress' || state === 'running'
}

/** React Query refetch policy: poll only while the backend says work is active. */
export function achievementsRefetchInterval(query: { state: { data: unknown } }): false | number {
  const data = query.state.data as AchievementsResponse | undefined

  return scanIsActive(data?.scan_meta) ? ACTIVE_SCAN_POLL_MS : false
}

/** A scan can fail after a stale snapshot has already been returned. The
 * canonical backend communicates that through scan_meta.status even when the
 * response's top-level error remains null. */
export function scanFailure(data?: AchievementsResponse): string | null {
  const status = data?.scan_meta.status
  const failed = status?.state?.toLowerCase() === 'failed'

  return data?.error || (failed ? status?.last_error || 'Scan failed.' : null)
}
