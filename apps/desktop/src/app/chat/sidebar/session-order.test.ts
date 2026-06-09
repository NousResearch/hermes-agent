import { describe, expect, it } from 'vitest'

import { topRecentSessions, workHeadSessionIds } from './session-order'

interface TimedRow {
  archived?: boolean
  id: string
  root?: null | string
  t: number
}

const timed = (...specs: Array<[string, number, (null | string | undefined)?, boolean?]>): TimedRow[] =>
  specs.map(([id, t, root, archived]) => ({ archived, id, root, t }))

const timedId = (item: TimedRow) => item.id
const timedRoot = (item: TimedRow) => item.root ?? null
const timedTime = (item: TimedRow) => item.t
const timedArchived = (item: TimedRow) => item.archived === true
const timedIds = (items: TimedRow[]) => items.map(item => item.id)
const sortedSetIds = (ids: Set<string>) => [...ids].sort()

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

describe('workHeadSessionIds', () => {
  it('returns only the freshest session id per lineage root', () => {
    const rows = timed(['root-a', 10, 'work-a'], ['tip-a', 30, 'work-a'], ['other', 20, 'work-b'])

    expect(sortedSetIds(workHeadSessionIds([rows], timedId, timedRoot, timedTime, timedArchived))).toEqual([
      'other',
      'tip-a'
    ])
  })

  it('treats sessions without a lineage root as their own open work item', () => {
    const rows = timed(['solo-a', 10], ['solo-b', 30])

    expect(sortedSetIds(workHeadSessionIds([rows], timedId, timedRoot, timedTime, timedArchived))).toEqual([
      'solo-a',
      'solo-b'
    ])
  })

  it('skips archived sessions and falls back to the newest unarchived tip in the lineage', () => {
    const rows = timed(['old-open', 10, 'work-a'], ['archived-tip', 99, 'work-a', true], ['fully-archived', 80, 'work-b', true])

    expect(sortedSetIds(workHeadSessionIds([rows], timedId, timedRoot, timedTime, timedArchived))).toEqual([
      'old-open'
    ])
  })

  it('dedupes ids across several lists while still choosing the freshest lineage head', () => {
    const recents = timed(['work-tip', 20, 'work-a'], ['cron-copy', 15, 'work-b'])
    const cron = timed(['work-tip', 20, 'work-a'], ['cron-newer', 60, 'work-b'])

    expect(sortedSetIds(workHeadSessionIds([recents, cron], timedId, timedRoot, timedTime, timedArchived))).toEqual([
      'cron-newer',
      'work-tip'
    ])
  })

  it('returns an empty set when no unarchived sessions exist', () => {
    expect(workHeadSessionIds([], timedId, timedRoot, timedTime, timedArchived).size).toBe(0)
    expect(workHeadSessionIds([timed(['archived', 1, 'work', true])], timedId, timedRoot, timedTime, timedArchived).size).toBe(0)
  })
})
