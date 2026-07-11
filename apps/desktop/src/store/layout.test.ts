import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { SessionInfo } from '@/types/hermes'

import {
  $pinnedSessionIds,
  $sidebarPinnedOrderIds,
  migrateLegacyPinnedSessions,
  setSidebarPinnedOrderIds,
  SIDEBAR_PINNED_STORAGE_KEY
} from './layout'
import { setSessions } from './session'

const session = (over: Partial<SessionInfo>): SessionInfo => ({
  archived: false,
  cwd: null,
  ended_at: null,
  id: 'live',
  input_tokens: 0,
  is_active: false,
  last_active: 0,
  message_count: 0,
  model: null,
  output_tokens: 0,
  preview: null,
  source: null,
  started_at: 0,
  title: null,
  tool_call_count: 0,
  ...over
})

describe('server-side pinned sessions', () => {
  beforeEach(() => {
    window.localStorage.clear()
    setSessions([])
  })

  it('derives pinned ids from server session data using lineage roots', () => {
    window.localStorage.setItem(SIDEBAR_PINNED_STORAGE_KEY, JSON.stringify(['stale-local']))

    setSessions([
      session({ id: 'recent', pinned: false }),
      session({ id: 'tip', _lineage_root_id: 'root', pinned: true })
    ])

    expect($pinnedSessionIds.get()).toEqual(['root'])
  })

  it('migrates legacy local pins as additive pushes and clears the old key', async () => {
    window.localStorage.setItem(SIDEBAR_PINNED_STORAGE_KEY, JSON.stringify(['already', 'legacy', 'legacy']))
    const pushed: string[] = []
    const log = { info: vi.fn() }

    await expect(
      migrateLegacyPinnedSessions([session({ id: 'already', pinned: true })], async id => {
        pushed.push(id)
      }, log)
    ).resolves.toBe(1)

    expect(pushed).toEqual(['legacy'])
    expect(window.localStorage.getItem(SIDEBAR_PINNED_STORAGE_KEY)).toBeNull()
    expect(log.info).toHaveBeenCalledWith('server-side pinned sessions migration push count', {
      pushed: 1,
      total: 3
    })
  })

  it('keeps the legacy key when a migration push fails so startup can retry', async () => {
    window.localStorage.setItem(SIDEBAR_PINNED_STORAGE_KEY, JSON.stringify(['legacy']))

    await expect(
      migrateLegacyPinnedSessions([], async () => {
        throw new Error('backend unavailable')
      })
    ).rejects.toThrow('backend unavailable')

    expect(window.localStorage.getItem(SIDEBAR_PINNED_STORAGE_KEY)).toBe(JSON.stringify(['legacy']))
  })

  it('keeps the local pinned drag-order independent of the server pin set', () => {
    // Pin membership (server-synced) and drag-order (local) are separate stores:
    // setting the order must not mutate the pinned set, and must dedupe/no-op on
    // an unchanged write.
    setSidebarPinnedOrderIds(['b', 'a', 'c'])
    expect($sidebarPinnedOrderIds.get()).toEqual(['b', 'a', 'c'])

    const ref = $sidebarPinnedOrderIds.get()
    setSidebarPinnedOrderIds(['b', 'a', 'c'])
    expect($sidebarPinnedOrderIds.get()).toBe(ref) // no-op keeps the same reference
  })
})
