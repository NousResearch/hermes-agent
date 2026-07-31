import { describe, expect, it, vi } from 'vitest'

import type { SessionInfo } from '@/hermes'

import {
  hydratePinnedSessions,
  MAX_PIN_HYDRATION_IDS,
  missingPinnedSessionIds,
  PIN_HYDRATION_CONCURRENCY,
  pinHydrationProfiles
} from './pinned-session-hydration'

const row = (id: string, extra: Partial<SessionInfo> = {}): SessionInfo =>
  ({ id, message_count: 1, source: 'desktop', started_at: 1, title: id, ...extra }) as SessionInfo

describe('hydratePinnedSessions', () => {
  it('hydrates pins outside the recent page in persisted pin order', async () => {
    const getSession = vi.fn(async (id: string, profile: null | string) => {
      if (profile !== 'default') {
        throw new Error('not found')
      }

      return row(id, { profile })
    })

    const hydrated = await hydratePinnedSessions(['old-pin', 'recent-pin'], ['default'], getSession)

    expect(hydrated.map(session => session.id)).toEqual(['old-pin', 'recent-pin'])
    expect(getSession).toHaveBeenCalledWith('old-pin', 'default')
  })

  it('falls through profiles and ignores missing or archived pins without failing the refresh', async () => {
    const getSession = vi.fn(async (id: string, profile: null | string) => {
      if (id === 'missing' || profile === 'default') {
        throw new Error('not found')
      }

      return row(id, { archived: id === 'archived', profile: profile ?? undefined })
    })

    const hydrated = await hydratePinnedSessions(['work-pin', 'missing', 'archived'], ['default', 'work'], getSession)

    expect(hydrated.map(session => session.id)).toEqual(['work-pin'])
    expect(getSession).toHaveBeenCalledWith('work-pin', 'work')
  })

  it('stamps the explicitly probed Desktop profile and preserves a historical local pin for migration', async () => {
    const hydrated = await hydratePinnedSessions(['remote-pin'], ['work'], async id =>
      row(id, { pinned: false, profile: 'default' })
    )

    expect(hydrated).toEqual([expect.objectContaining({ id: 'remote-pin', profile: 'work', pinned: undefined })])
  })

  it('bounds both the pin budget and concurrent profile probes', async () => {
    let active = 0
    let peak = 0
    let calls = 0
    const ids = Array.from({ length: MAX_PIN_HYDRATION_IDS + 25 }, (_, index) => `pin-${index}`)

    const hydrated = await hydratePinnedSessions(ids, ['default', 'work'], async (id, profile) => {
      calls += 1
      active += 1
      peak = Math.max(peak, active)
      await Promise.resolve()
      active -= 1

      if (profile === 'default') {
        throw new Error('not found')
      }

      return row(id, { profile: 'default' })
    })

    expect(hydrated).toHaveLength(MAX_PIN_HYDRATION_IDS)
    expect(calls).toBe(MAX_PIN_HYDRATION_IDS * 2)
    expect(peak).toBeLessThanOrEqual(PIN_HYDRATION_CONCURRENCY)
  })

  it('only hydrates pins missing from every already-loaded sidebar slice', () => {
    const loaded = [row('tip', { _lineage_root_id: 'root-pin' }), row('cron-pin')]

    expect(missingPinnedSessionIds(['root-pin', 'cron-pin', 'old-pin'], loaded)).toEqual(['old-pin'])
  })

  it('scopes hydration to one profile or all known profiles with default fallback', () => {
    expect(pinHydrationProfiles('work', ['default', 'work'])).toEqual(['work'])
    expect(pinHydrationProfiles('__all__', ['work', 'default', 'work'])).toEqual(['default', 'work'])
    expect(pinHydrationProfiles('__all__', [])).toEqual(['default'])
  })
})
