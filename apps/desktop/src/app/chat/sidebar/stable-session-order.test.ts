import { describe, expect, it } from 'vitest'

import type { SessionInfo } from '@/types/hermes'

import { stabilizeSessionOrder } from './stable-session-order'

const row = (id: string, lastActive: number, profile = 'default', root?: string): SessionInfo =>
  ({ id, last_active: lastActive, profile, started_at: lastActive, _lineage_root_id: root }) as SessionInfo

describe('stabilizeSessionOrder', () => {
  it('does not reshuffle existing rows when background activity changes recency', () => {
    const previous = ['default::a', 'default::b', 'default::c']
    const refreshed = [row('c', 30), row('a', 20), row('b', 10)]

    expect(stabilizeSessionOrder(previous, refreshed).sessions.map(session => session.id)).toEqual(['a', 'b', 'c'])
  })

  it('inserts genuinely new rows at their recency position without moving existing rows', () => {
    const previous = ['default::a', 'default::b', 'default::c']
    const refreshed = [row('new-top', 50), row('a', 40), row('b', 30), row('new-middle', 20), row('c', 10)]

    expect(stabilizeSessionOrder(previous, refreshed).sessions.map(session => session.id)).toEqual([
      'new-top',
      'a',
      'b',
      'new-middle',
      'c'
    ])
  })

  it('promotes only the explicitly selected conversation', () => {
    const previous = ['default::a', 'default::b', 'default::c']
    const refreshed = [row('c', 30), row('b', 20), row('a', 10)]

    expect(stabilizeSessionOrder(previous, refreshed, 'c').sessions.map(session => session.id)).toEqual(['c', 'a', 'b'])
  })

  it('keeps one stable key across compression tip rotation', () => {
    const previous = ['default::root']
    const refreshed = [row('tip-2', 20, 'default', 'root')]
    const result = stabilizeSessionOrder(previous, refreshed)

    expect(result.keys).toEqual(['default::root'])
    expect(result.sessions.map(session => session.id)).toEqual(['tip-2'])
  })
})
