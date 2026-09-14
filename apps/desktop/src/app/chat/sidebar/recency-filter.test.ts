import { describe, expect, it } from 'vitest'

import type { SessionInfo } from '@/hermes'

import { sessionMatchesRecencyFilter } from './recency-filter'

const NOW = 1_800_000_000
const HOUR = 60 * 60
const DAY = 24 * HOUR

const session = (fields: { last_active?: number; started_at?: number }) => fields as unknown as SessionInfo

describe('sessionMatchesRecencyFilter', () => {
  it('matches everything when no window is selected', () => {
    expect(sessionMatchesRecencyFilter(session({ last_active: NOW - 10 * DAY }), [], NOW)).toBe(true)
  })

  it('narrows to the last 24 hours for "1 day"', () => {
    expect(sessionMatchesRecencyFilter(session({ last_active: NOW - 23 * HOUR }), ['1d'], NOW)).toBe(true)
    expect(sessionMatchesRecencyFilter(session({ last_active: NOW - 25 * HOUR }), ['1d'], NOW)).toBe(false)
  })

  it('narrows to the last 48 hours for "2 day"', () => {
    expect(sessionMatchesRecencyFilter(session({ last_active: NOW - 47 * HOUR }), ['2d'], NOW)).toBe(true)
    expect(sessionMatchesRecencyFilter(session({ last_active: NOW - 49 * HOUR }), ['2d'], NOW)).toBe(false)
  })

  it('is a union across windows — both selected behaves as the 48-hour one', () => {
    const withinTwoDays = session({ last_active: NOW - 25 * HOUR })

    expect(sessionMatchesRecencyFilter(withinTwoDays, ['1d', '2d'], NOW)).toBe(true)
    expect(sessionMatchesRecencyFilter(withinTwoDays, ['1d'], NOW)).toBe(false)
  })

  it('keeps the 24-hour boundary inclusive', () => {
    expect(sessionMatchesRecencyFilter(session({ last_active: NOW - DAY }), ['1d'], NOW)).toBe(true)
    expect(sessionMatchesRecencyFilter(session({ last_active: NOW - DAY - 1 }), ['1d'], NOW)).toBe(false)
  })

  it('falls back to started_at when last_active was never stamped', () => {
    const fresh = session({ last_active: 0, started_at: NOW - 2 * HOUR })

    expect(sessionMatchesRecencyFilter(fresh, ['1d'], NOW)).toBe(true)
  })

  it('excludes a session with no timestamps at all', () => {
    expect(sessionMatchesRecencyFilter(session({}), ['1d'], NOW)).toBe(false)
  })
})
