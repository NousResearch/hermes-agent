import { describe, expect, it } from 'vitest'

import { freshestSessionId, topRecentSessions } from './session-order'

interface TimedRow {
  id: string
  t: number
}

const timed = (...specs: Array<[string, number]>): TimedRow[] => specs.map(([id, t]) => ({ id, t }))
const timedId = (item: TimedRow) => item.id
const timedTime = (item: TimedRow) => item.t
const timedIds = (items: TimedRow[]) => items.map(item => item.id)

describe('topRecentSessions', () => {
  it('merges several lists, sorts newest-first, and caps at n', () => {
    const local = timed(['a', 10], ['b', 30])
    const slack = timed(['s', 45])
    const cron = timed(['c', 20], ['d', 5])

    expect(timedIds(topRecentSessions([local, slack, cron], timedId, timedTime, 3))).toEqual(['s', 'b', 'c'])
  })

  it('dedupes by id, keeping the freshest timestamp on collision', () => {
    const loaded = timed(['x', 10], ['y', 40])
    const cron = timed(['x', 99]) // same session id, but fresher

    expect(timedIds(topRecentSessions([loaded, cron], timedId, timedTime, 5))).toEqual(['x', 'y'])
  })

  it('returns an empty list for n <= 0 or no input', () => {
    expect(topRecentSessions([], timedId, timedTime, 5)).toEqual([])
    expect(topRecentSessions([timed(['a', 1])], timedId, timedTime, 0)).toEqual([])
  })
})

describe('freshestSessionId', () => {
  it('returns the id of the single newest session across lists', () => {
    expect(freshestSessionId([timed(['a', 10]), timed(['b', 50], ['c', 5])], timedId, timedTime)).toBe('b')
  })

  it('returns null when there are no sessions', () => {
    expect(freshestSessionId([], timedId, timedTime)).toBeNull()
  })
})
