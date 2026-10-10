import { describe, expect, it } from 'vitest'

import type { SessionInfo } from '@/types/hermes'

import { buildSessionByAnyId, compareSessionTitles, resolvePinnedSessions } from './session-index'

const row = (id: string, extra: Partial<SessionInfo> = {}): SessionInfo =>
  ({ id, message_count: 1, source: 'cli', started_at: 0, title: id, ...extra }) as SessionInfo

// no pin write in flight - the server flag is trustworthy.
const settled: ReadonlySet<string> = new Set()

describe('buildSessionByAnyId', () => {
  it('resolves a pin from every slice the sidebar fetches', () => {
    const index = buildSessionByAnyId([row('recent')], [row('cron_job_1')], [row('telegram_42')])

    for (const id of ['recent', 'cron_job_1', 'telegram_42']) {
      expect(index.get(id)?.id).toBe(id)
    }
  })

  it('resolves a pin stored on the pre-compression lineage root', () => {
    const index = buildSessionByAnyId([], [], [row('tip', { _lineage_root_id: 'root' })])

    expect(index.get('root')?.id).toBe('tip')
    expect(index.get('tip')?.id).toBe('tip')
  })

  it('lets a recents row win a direct id collision', () => {
    const index = buildSessionByAnyId(
      [row('dupe', { title: 'from recents' })],
      [],
      [row('dupe', { title: 'from messaging' })]
    )

    expect(index.get('dupe')?.title).toBe('from recents')
  })

  it('does not let a lineage alias clobber a real row under that id', () => {
    const index = buildSessionByAnyId([row('root')], [], [row('tip', { _lineage_root_id: 'root' })])

    expect(index.get('root')?.id).toBe('root')
  })
})

describe('compareSessionTitles', () => {
  it('sorts alphabetically in case-insensitive order', () => {
    const s1 = row('1', { title: 'apple' })
    const s2 = row('2', { title: 'Banana' })
    const s3 = row('3', { title: 'cherry' })

    const list = [s2, s3, s1].sort(compareSessionTitles)
    expect(list.map(s => s.title)).toEqual(['apple', 'Banana', 'cherry'])
  })

  it('sorts portuguese and accented characters naturally without separate blocks', () => {
    const s1 = row('1', { title: 'árvore' })
    const s2 = row('2', { title: 'banana' })
    const s3 = row('3', { title: 'coração' })
    const s4 = row('4', { title: 'dado' })

    const list = [s3, s1, s4, s2].sort(compareSessionTitles)
    expect(list.map(s => s.title)).toEqual(['árvore', 'banana', 'coração', 'dado'])
  })

  it('sorts numbered titles naturally using numeric collation', () => {
    const s1 = row('1', { title: 'session 1' })
    const s2 = row('2', { title: 'session 2' })
    const s10 = row('10', { title: 'session 10' })

    const list = [s10, s2, s1].sort(compareSessionTitles)
    expect(list.map(s => s.title)).toEqual(['session 1', 'session 2', 'session 10'])
  })

  it('falls back to preview when title is missing', () => {
    const s1 = row('1', { preview: 'alpha preview', title: '' })
    const s2 = row('2', { preview: 'beta preview', title: null as unknown as string })

    const list = [s2, s1].sort(compareSessionTitles)
    expect(list.map(s => s.id)).toEqual(['1', '2'])
  })

  it('normalizes whitespace-only titles to fall back to preview or untitled session', () => {
    const s1 = row('1', { preview: 'first preview', title: '   ' })
    const s2 = row('2', { preview: 'second preview', title: '\t\n' })

    const list = [s2, s1].sort(compareSessionTitles)
    expect(list.map(s => s.id)).toEqual(['1', '2'])
  })

  it('breaks ties deterministically using session id', () => {
    const s1 = row('id-b', { title: 'same title' })
    const s2 = row('id-a', { title: 'same title' })

    const list = [s1, s2].sort(compareSessionTitles)
    expect(list.map(s => s.id)).toEqual(['id-a', 'id-b'])
  })
})

describe('resolvePinnedSessions', () => {
  it('sorts resolved pinned sessions alphabetically by title', () => {
    const sessions = [row('c', { title: 'zebra' }), row('b', { title: 'banana' }), row('a', { title: 'apple' })]
    const index = buildSessionByAnyId(sessions, [], [])

    expect(resolvePinnedSessions(['c', 'a', 'b'], index, sessions, settled).map(s => s.title)).toEqual([
      'apple',
      'banana',
      'zebra'
    ])
  })

  it('falls back to the server pinned flag when local storage is cold', () => {
    const sessions = [row('b', { pinned: true, title: 'zebra' }), row('a', { pinned: true, title: 'apple' })]
    const index = buildSessionByAnyId(sessions, [], [])

    expect(resolvePinnedSessions([], index, sessions, settled).map(s => s.title)).toEqual(['apple', 'zebra'])
  })

  it('does not duplicate a session held both locally and server-side', () => {
    const sessions = [row('a', { pinned: true, title: 'apple' })]
    const index = buildSessionByAnyId(sessions, [], [])

    expect(resolvePinnedSessions(['a'], index, sessions, settled).map(s => s.id)).toEqual(['a'])
  })

  it('does not duplicate a server-pinned row whose pin is stored on the lineage root', () => {
    const sessions = [row('tip', { _lineage_root_id: 'root', pinned: true, title: 'apple' })]
    const index = buildSessionByAnyId(sessions, [], [])

    expect(resolvePinnedSessions(['root'], index, sessions, settled).map(s => s.id)).toEqual(['tip'])
  })

  it('combines local and server pins and sorts them alphabetically by title', () => {
    const sessions = [row('server-pin', { pinned: true, title: 'apple' }), row('local-pin', { title: 'zebra' })]
    const index = buildSessionByAnyId(sessions, [], [])

    expect(resolvePinnedSessions(['local-pin'], index, sessions, settled).map(s => s.id)).toEqual([
      'server-pin',
      'local-pin'
    ])
  })

  it('does not resurrect a session the user just unpinned', () => {
    const sessions = [row('just-unpinned', { pinned: true })]
    const index = buildSessionByAnyId(sessions, [], [])

    expect(resolvePinnedSessions([], index, sessions, new Set(['just-unpinned']))).toEqual([])
  })

  it('fences a stale row under the lineage root the pin was written on', () => {
    const sessions = [row('tip', { _lineage_root_id: 'root', pinned: true })]
    const index = buildSessionByAnyId(sessions, [], [])

    expect(resolvePinnedSessions([], index, sessions, new Set(['root']))).toEqual([])
  })

  it('still adopts a foreign pin while an unrelated write is in flight', () => {
    const sessions = [row('foreign', { pinned: true })]
    const index = buildSessionByAnyId(sessions, [], [])

    expect(resolvePinnedSessions([], index, sessions, new Set(['other'])).map(s => s.id)).toEqual(['foreign'])
  })

  it('sorts consistently across activity changes and profile switches', () => {
    const sessions = [
      row('foreign', { last_active: 1, pinned: true, profile: 'k9', title: 'zebra' }),
      row('local', { last_active: 50, pinned: true, profile: 'default', title: 'apple' })
    ]

    const index = buildSessionByAnyId(sessions, [], [])

    expect(resolvePinnedSessions(['foreign', 'local'], index, sessions, settled).map(s => s.id)).toEqual([
      'local',
      'foreign'
    ])

    const clicked = [
      row('foreign', { last_active: 99, pinned: true, profile: 'k9', title: 'zebra' }),
      row('local', { last_active: 50, pinned: true, profile: 'default', title: 'apple' })
    ]

    expect(resolvePinnedSessions(['foreign', 'local'], index, clicked, settled).map(s => s.id)).toEqual([
      'local',
      'foreign'
    ])
  })

  it('ignores rows from a backend that predates the pinned flag', () => {
    const sessions = [row('a')]
    const index = buildSessionByAnyId(sessions, [], [])

    expect(resolvePinnedSessions([], index, sessions, settled)).toEqual([])
  })
})
