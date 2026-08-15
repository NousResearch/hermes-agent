import { describe, expect, it } from 'vitest'

import type { SessionInfo } from '@/types/hermes'

import { buildSessionByAnyId, resolvePinnedSessions } from './session-index'

const row = (id: string, extra: Partial<SessionInfo> = {}): SessionInfo =>
  ({ id, message_count: 1, source: 'cli', started_at: 0, title: id, ...extra }) as SessionInfo

describe('buildSessionByAnyId', () => {
  it('resolves a pin from every slice the sidebar fetches', () => {
    // The contract that matters: a pin is looked up in this map no matter
    // which slice owns the row. Messaging is the one that regressed — a
    // pinned session is filtered out of its own section, so a miss here
    // removes it from the sidebar entirely rather than just misplacing it.
    const index = buildSessionByAnyId([row('recent')], [row('cron_job_1')], [row('telegram_42')])

    for (const id of ['recent', 'cron_job_1', 'telegram_42']) {
      expect(index.get(id)?.id).toBe(id)
    }
  })

  it('resolves a pin stored on the pre-compression lineage root', () => {
    const index = buildSessionByAnyId([], [], [row('tip', { _lineage_root_id: 'root' })])

    // Both identities point at the one live row.
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
    // 'root' is a live session in its own right AND another row's lineage
    // root; the real row must win so the pin opens the right conversation.
    const index = buildSessionByAnyId([row('root')], [], [row('tip', { _lineage_root_id: 'root' })])

    expect(index.get('root')?.id).toBe('root')
  })
})

describe('resolvePinnedSessions', () => {
  it('resolves local pin ids in their hand-picked order', () => {
    const sessions = [row('a'), row('b'), row('c')]
    const index = buildSessionByAnyId(sessions, [], [])

    expect(resolvePinnedSessions(['c', 'a'], index, sessions).map(s => s.id)).toEqual(['c', 'a'])
  })

  it('falls back to the server pinned flag when localStorage is cold (#85969)', () => {
    // Backend says pinned=1 but the local pin set is empty (cold localStorage
    // after a reload, pin from another client, clobbered persist). Every other
    // list filters the row out as pinned, so if the Pinned section can't
    // resolve it the session vanishes from the sidebar entirely.
    const sessions = [row('a', { pinned: true }), row('b', { pinned: false })]
    const index = buildSessionByAnyId(sessions, [], [])

    expect(resolvePinnedSessions([], index, sessions).map(s => s.id)).toEqual(['a'])
  })

  it('does not duplicate a session held both locally and server-side', () => {
    const sessions = [row('a', { pinned: true })]
    const index = buildSessionByAnyId(sessions, [], [])

    expect(resolvePinnedSessions(['a'], index, sessions).map(s => s.id)).toEqual(['a'])
  })

  it('does not duplicate a server-pinned row whose pin is stored on the lineage root', () => {
    const sessions = [row('tip', { _lineage_root_id: 'root', pinned: true })]
    const index = buildSessionByAnyId(sessions, [], [])

    expect(resolvePinnedSessions(['root'], index, sessions).map(s => s.id)).toEqual(['tip'])
  })

  it('keeps locally pinned rows ahead of server-only fallback pins', () => {
    const sessions = [row('server-pin', { pinned: true }), row('local-pin')]
    const index = buildSessionByAnyId(sessions, [], [])

    expect(resolvePinnedSessions(['local-pin'], index, sessions).map(s => s.id)).toEqual(['local-pin', 'server-pin'])
  })

  it('ignores rows from a backend that predates the pinned flag', () => {
    // `pinned` undefined means "no opinion", never "pinned".
    const sessions = [row('a')]
    const index = buildSessionByAnyId(sessions, [], [])

    expect(resolvePinnedSessions([], index, sessions)).toEqual([])
  })
})
