/**
 * Harvest bound for stranded group replies (#105247): the background watch
 * must cover the whole legal life of a turn (hard cap + margin), or a reply
 * landing in a quiet room after the old fixed 5-minute window was dropped.
 */
import { describe, expect, it, vi } from 'vitest'

vi.mock('@hermes/plugin-sdk', async () => {
  const { atom } = await import('nanostores')

  return {
    atom,
    host: { state: { connectionId: { get: () => 'local' } } }
  }
})

import { GROUP_TURN_HARD_CAP_MS, groupStrandedHarvestMaxTries } from './group-turns'

describe('groupStrandedHarvestMaxTries', () => {
  it('covers the hard cap plus a margin at the production interval', () => {
    const tries = groupStrandedHarvestMaxTries(5000)

    expect(tries * 5000).toBeGreaterThanOrEqual(GROUP_TURN_HARD_CAP_MS + 60000)
    // 20min cap / 5s + 60s margin = 252 tries (~21min), not the old 60 (5min).
    expect(tries).toBeGreaterThan(60)
  })

  it('keeps the old floor for degenerate intervals', () => {
    expect(groupStrandedHarvestMaxTries(Number.MAX_SAFE_INTEGER)).toBe(60)
  })

  it('scales with the interval instead of hardcoding tries', () => {
    const fast = groupStrandedHarvestMaxTries(1000)
    const slow = groupStrandedHarvestMaxTries(30000)

    expect(fast * 1000).toBeGreaterThanOrEqual(GROUP_TURN_HARD_CAP_MS + 60000)
    expect(slow * 30000).toBeGreaterThanOrEqual(GROUP_TURN_HARD_CAP_MS + 60000)
    expect(fast).toBeGreaterThan(slow)
  })
})
