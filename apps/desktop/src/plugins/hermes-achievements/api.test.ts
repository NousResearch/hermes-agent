import { afterEach, describe, expect, it, vi } from 'vitest'

import {
  ACTIVE_SCAN_POLL_MS,
  achievementsRefetchInterval,
  bindAchievementsApi,
  fetchAchievements,
  rescanAchievements,
  scanFailure,
  scanIsActive
} from './api'
import type { AchievementsResponse } from './types'

const payload = (overrides: Partial<AchievementsResponse> = {}): AchievementsResponse => ({
  achievements: [],
  discovered_count: 0,
  error: null,
  generated_at: 1,
  is_stale: false,
  scan_meta: { mode: 'incremental', status: { state: 'idle' } },
  secret_count: 0,
  total_count: 0,
  unlocked_count: 0,
  ...overrides
})

let dispose: (() => void) | undefined

afterEach(() => {
  dispose?.()
  dispose = undefined
})

describe('achievements API boundary', () => {
  it('uses only the bound plugin REST namespace for reads and rescans', async () => {
    const rest = vi.fn(async <T>() => payload() as T)
    dispose = bindAchievementsApi(rest)

    await fetchAchievements()
    await expect(rescanAchievements()).resolves.toMatchObject({ is_stale: false })

    expect(rest).toHaveBeenNthCalledWith(1, '/achievements')
    expect(rest).toHaveBeenNthCalledWith(2, '/rescan', { method: 'POST' })
  })

  it('rejects after the plugin is disposed instead of retaining a stale REST door', async () => {
    dispose = bindAchievementsApi(vi.fn(async <T>() => undefined as T))
    dispose()
    dispose = undefined

    await expect(fetchAchievements()).rejects.toThrow('achievements api not ready')
  })
})

describe('scan polling policy', () => {
  it.each([
    [{ mode: 'pending' }, true],
    [{ mode: 'in_progress' }, true],
    [{ mode: 'incremental', status: { state: 'running' } }, true],
    [{ mode: 'full', status: { state: 'idle' } }, false],
    [{ mode: 'failed', status: { state: 'failed' } }, false]
  ])('recognizes active state from mode or nested status: %o', (scanMeta, expected) => {
    expect(scanIsActive(scanMeta)).toBe(expected)
  })

  it('polls GET /achievements only while scanning', () => {
    expect(achievementsRefetchInterval({ state: { data: payload({ scan_meta: { mode: 'pending' } }) } })).toBe(
      ACTIVE_SCAN_POLL_MS
    )
    expect(
      achievementsRefetchInterval({ state: { data: payload({ scan_meta: { status: { state: 'idle' } } }) } })
    ).toBe(false)
  })

  it('surfaces nested scan failure when the top-level error is absent', () => {
    expect(
      scanFailure(
        payload({ error: null, scan_meta: { status: { last_error: 'database unavailable', state: 'failed' } } })
      )
    ).toBe('database unavailable')
    expect(scanFailure(payload({ error: 'top-level', scan_meta: { status: { last_error: 'nested', state: 'failed' } } }))).toBe(
      'top-level'
    )
  })
})
