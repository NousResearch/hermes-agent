import { describe, expect, it } from 'vitest'

import { makeSessionInfo } from '@/test/session-info'

import { buildMessagingSections, splitArchivedSessions } from './sidebar-messaging'

const row = (id: string, source: null | string, last_active = 0) =>
  makeSessionInfo({ archived: true, id, last_active, source })

const ids = (rows: readonly { id: string }[]) => rows.map(r => r.id)

const notPinned = () => false

describe('splitArchivedSessions', () => {
  it('routes every archived row to exactly one list, messaging threads to their platforms', () => {
    const archived = [
      row('tg', 'telegram'),
      row('desk', 'desktop'),
      row('dc', ' Discord '),
      row('cli', 'cli'),
      row('none', null)
    ]

    const { messaging, sessions } = splitArchivedSessions(archived)

    expect(ids(messaging)).toEqual(['tg', 'dc'])
    expect(ids(sessions)).toEqual(['desk', 'cli', 'none'])
    expect([...ids(messaging), ...ids(sessions)].sort()).toEqual(ids(archived).sort())
  })
})

describe('buildMessagingSections', () => {
  const threads = [row('tg-old', 'telegram', 1), row('dc', 'discord', 5), row('tg-new', 'telegram', 9)]

  it('groups by platform, newest platform and newest row first, leaving pinned rows to the Pinned section', () => {
    const sections = buildMessagingSections([...threads, row('tg-pin', 'telegram', 99)], r => r.id === 'tg-pin', null)

    expect(sections.map(s => [s.sourceId, ids(s.sessions)])).toEqual([
      ['telegram', ['tg-new', 'tg-old']],
      ['discord', ['dc']]
    ])
  })

  it('offers neither "load more" nor a backend total when the rows are the complete set', () => {
    const [telegram] = buildMessagingSections(threads, notPinned, null)

    expect(telegram).toMatchObject({ hasMore: false, total: 2 })
  })

  it('pages a live platform by its known total, discounted by pins, or by the seed cap until one resolves', () => {
    const totals: Record<string, number> = { telegram: 4 }
    const paging = { totalFor: (sourceId: string) => totals[sourceId], truncated: true }

    const [telegram, discord] = buildMessagingSections(
      [...threads, row('tg-pin', 'telegram', 0)],
      r => r.id === 'tg-pin',
      paging
    )

    expect(telegram).toMatchObject({ hasMore: true, total: 3 })
    expect(discord).toMatchObject({ hasMore: true, total: 1 })
    expect(buildMessagingSections(threads, notPinned, { ...paging, truncated: false })[1].hasMore).toBe(false)
  })
})
