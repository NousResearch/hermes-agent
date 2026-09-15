import { beforeEach, describe, expect, it } from 'vitest'

import { $cronSessions, $messagingSessions, $unlistedSessionOwnerRows, setSessions } from '@/store/session'
import { $removedSessionIds } from '@/store/session-removal'
import { $sessionTiles } from '@/store/session-states'
import type { SessionInfo } from '@/types/hermes'

import { makeSessionInfo } from '../test/session-info'

import { $liveSessions, $visibleLiveSessions, clearLiveSessions, reconcileLiveSessions } from './live-sessions'

/**
 * `session.active_list` is the only thing that can show a session created over
 * the gateway by ANOTHER client before its first prompt persists a DB row —
 * the stored sidebar slice filters `min_message_count=1`, so the row can never
 * appear there until then (#50799). The live group must therefore be fed by
 * the existing poll without ever duplicating a session the sidebar (or the
 * owner ladder) already represents.
 */

const OPTS = { connectionId: 'conn-1', profileKey: 'workops' } as const

function liveItem(overrides: Record<string, unknown> = {}) {
  return {
    id: 'runtime-1',
    last_active: 2_000,
    message_count: 0,
    model: 'zyphra/qwen',
    preview: 'first prompt',
    session_key: 'sess-live-a',
    source: 'cli',
    started_at: 1_000,
    status: 'working',
    title: 'Live chat',
    ...overrides
  }
}

const rowById = (id: string): SessionInfo | undefined => $liveSessions.get().find(row => row.id === id)

beforeEach(() => {
  // Real atoms, controlled directly — reset every store the reconciler reads
  // so cross-test state can't leak (atoms persist module-wide).
  clearLiveSessions()
  $liveSessions.set([])
  setSessions([])
  $cronSessions.set([])
  $messagingSessions.set([])
  $unlistedSessionOwnerRows.set([])
  $removedSessionIds.set(new Set())
  $sessionTiles.set([])
})

describe('reconcileLiveSessions — live rows for unpersisted sessions', () => {
  it('turns a live unpersisted session into exactly one stamped row', () => {
    const rows = reconcileLiveSessions({ sessions: [liveItem()] }, OPTS)

    expect(rows).toHaveLength(1)
    const row = rows[0]

    // The sidebar/owner ladder keys on the STORED id: the live session_key,
    // never the ephemeral runtime id.
    expect(row.id).toBe('sess-live-a')
    // Owner resolution reads these stamps; a missing connection_id routes
    // session-scoped RPCs to the wrong backend (#102792 class).
    expect(row.connection_id).toBe('conn-1')
    expect(row.profile).toBe('workops')
    expect(row.is_default_profile).toBe(false)
    expect(row.title).toBe('Live chat')
    expect(row.preview).toBe('first prompt')
    expect(row.message_count).toBe(0)
    expect(row.model).toBe('zyphra/qwen')
    expect(row.started_at).toBe(1_000)
    expect(row.last_active).toBe(2_000)
    expect(row.ended_at).toBeNull()
    expect(row.input_tokens).toBe(0)
    expect(row.output_tokens).toBe(0)
    expect(row.tool_call_count).toBe(0)
    expect(row.is_active).toBe(true)
    expect(row.cwd).toBeNull()
  })

  it('falls back to the default profile flag when the poll profile is blank', () => {
    const rows = reconcileLiveSessions({ sessions: [liveItem()] }, { connectionId: 'c', profileKey: '  ' })

    expect(rows[0].profile).toBe('default')
    expect(rows[0].is_default_profile).toBe(true)
  })

  it('carries the per-item profile from the snapshot over the poll default', () => {
    const rows = reconcileLiveSessions({ sessions: [liveItem({ profile: 'backend-x' })] }, OPTS)

    expect(rows[0].profile).toBe('backend-x')
  })
})

describe('reconcileLiveSessions — dedupe against everything already represented', () => {
  const seeds: Array<[string, () => void]> = [
    ['a stored recents row', () => setSessions([makeSessionInfo({ id: 'sess-live-a' })])],
    [
      'a compression tip whose lineage contains the key',
      () =>
        setSessions([
          makeSessionInfo({ id: 'tip-9', _lineage_ids: ['sess-live-a', 'tip-9'], _lineage_root_id: 'sess-live-a' })
        ])
    ],
    ['a cron-slice row', () => $cronSessions.set([makeSessionInfo({ id: 'sess-live-a' })])],
    ['a messaging-slice row', () => $messagingSessions.set([makeSessionInfo({ id: 'sess-live-a' })])],
    ['an unlisted-draft owner stub', () => $unlistedSessionOwnerRows.set([makeSessionInfo({ id: 'sess-live-a' })])],
    ['a delete/archive tombstone', () => $removedSessionIds.set(new Set(['sess-live-a']))]
  ]

  it.each(seeds)('does not duplicate a live session already represented by %s', (_label, seed) => {
    seed()

    // The duplicate is dropped, a genuinely-unlisted live session survives —
    // so the emptiness assertion can't pass vacuously.
    const rows = reconcileLiveSessions({ sessions: [liveItem(), liveItem({ session_key: 'other', id: 'r2' })] }, OPTS)

    expect(rows.map(r => r.id)).toEqual(['other'])
  })

  it('dedupes duplicate session_keys within one snapshot to one row', () => {
    const rows = reconcileLiveSessions(
      { sessions: [liveItem(), liveItem({ id: 'runtime-2' }), liveItem({ session_key: ' sess-live-a ' })] },
      OPTS
    )

    expect(rows.map(r => r.id)).toEqual(['sess-live-a'])
  })

  it('drops items with no session_key — there is nothing to resume or dedupe', () => {
    const rows = reconcileLiveSessions(
      { sessions: [liveItem({ session_key: '   ' }), liveItem({ session_key: 'ok' })] },
      OPTS
    )

    expect(rows.map(r => r.id)).toEqual(['ok'])
  })

  it('keeps the row of a session that is open as a TILE — a tile is not a stored row', () => {
    // Clicking a live row opens `in-place`, which loads main (never a tile), so
    // a tile can only come from an explicit "open in tab". The session still has
    // no stored row in that state, so hiding it here would make the group look
    // like it ate the session; stored rows whose tile is open stay listed too.
    $sessionTiles.set([{ storedSessionId: 'sess-live-a' }])

    expect(reconcileLiveSessions({ sessions: [liveItem()] }, OPTS).map(r => r.id)).toEqual(['sess-live-a'])
  })

  it('keeps a live session that a list refresh has not returned YET, and drops it once the row lands', () => {
    expect(reconcileLiveSessions({ sessions: [liveItem()] }, OPTS)).toHaveLength(1)

    // The first prompt persisted the row and the sidebar slice came back with it.
    setSessions([makeSessionInfo({ id: 'sess-live-a', message_count: 1 })])
    reconcileLiveSessions({ sessions: [liveItem({ message_count: 1 })] }, OPTS)

    expect($liveSessions.get()).toEqual([])
  })
})

describe('reconcileLiveSessions — exclusions', () => {
  it('drops hidden items', () => {
    const rows = reconcileLiveSessions({ sessions: [liveItem({ hidden: true }), liveItem({ session_key: 'b' })] }, OPTS)

    expect(rows.map(r => r.id)).toEqual(['b'])
  })

  it.each(['cron', 'kanban', 'subagent', 'tool', 'telegram', 'discord', 'api_server'])(
    'drops source=%s — the live group obeys the sidebar exclusion set',
    source => {
      const rows = reconcileLiveSessions({ sessions: [liveItem({ source })] }, OPTS)

      expect(rows).toEqual([])
    }
  )

  it('includes an item with NO source (unknown on an older backend — the #50799 target case)', () => {
    const withoutKey = { ...liveItem() }

    delete (withoutKey as Record<string, unknown>).source

    const rows = reconcileLiveSessions({ sessions: [withoutKey] }, OPTS)

    expect(rows.map(r => r.id)).toEqual(['sess-live-a'])
    expect(rows[0].source).toBeNull()
  })

  it('normalizes the source onto the row', () => {
    const rows = reconcileLiveSessions({ sessions: [liveItem({ source: ' CLI ' })] }, OPTS)

    expect(rows[0].source).toBe('cli')
  })
})

describe('reconcileLiveSessions — no-information vs authoritative-empty', () => {
  it('leaves the atom untouched when sessions is undefined (failed/degraded request)', () => {
    reconcileLiveSessions({ sessions: [liveItem()] }, OPTS)
    const before = $liveSessions.get()

    expect(reconcileLiveSessions({}, OPTS)).toBe(before)
    expect(reconcileLiveSessions({ sessions: undefined }, OPTS)).toBe(before)
    expect($liveSessions.get()).toHaveLength(1)
  })

  it('prunes live sessions that left the snapshot, and clears on []', () => {
    reconcileLiveSessions({ sessions: [liveItem(), liveItem({ session_key: 'gone', id: 'runtime-2' })] }, OPTS)
    expect($liveSessions.get()).toHaveLength(2)

    // The gateway reaped `gone`; absence in an explicit snapshot is information.
    reconcileLiveSessions({ sessions: [liveItem()] }, OPTS)
    expect($liveSessions.get().map(r => r.id)).toEqual(['sess-live-a'])

    reconcileLiveSessions({ sessions: [] }, OPTS)
    expect($liveSessions.get()).toEqual([])
  })
})

describe('reconcileLiveSessions — identity and ordering', () => {
  it('preserves array identity when a no-op reconcile changes nothing', () => {
    reconcileLiveSessions({ sessions: [liveItem()] }, OPTS)
    const first = $liveSessions.get()

    const again = reconcileLiveSessions({ sessions: [liveItem()] }, OPTS)

    expect(again).toBe(first)
    expect($liveSessions.get()).toBe(first)
  })

  it('replaces the reference when a field actually changes', () => {
    reconcileLiveSessions({ sessions: [liveItem()] }, OPTS)
    const first = $liveSessions.get()

    const changed = reconcileLiveSessions({ sessions: [liveItem({ title: 'Renamed' })] }, OPTS)

    expect(changed).not.toBe(first)
    expect(rowById('sess-live-a')?.title).toBe('Renamed')
  })

  it('orders rows by last_active desc', () => {
    const rows = reconcileLiveSessions(
      {
        sessions: [
          liveItem({ session_key: 'old', id: 'r1', last_active: 1_500 }),
          liveItem({ session_key: 'newest', id: 'r2', last_active: 3_000 }),
          liveItem({ session_key: 'mid', id: 'r3', last_active: 2_000 })
        ]
      },
      OPTS
    )

    expect(rows.map(r => r.id)).toEqual(['newest', 'mid', 'old'])
  })

  it('clearLiveSessions empties the group for a gateway switch', () => {
    reconcileLiveSessions({ sessions: [liveItem()] }, OPTS)
    clearLiveSessions()
    expect($liveSessions.get()).toEqual([])
  })
})

describe('$visibleLiveSessions — promotion is read-time, not poll-time', () => {
  it('drops a live row the instant its stored row lands, before the next poll', () => {
    reconcileLiveSessions({ sessions: [liveItem()] }, OPTS)
    expect($visibleLiveSessions.get().map(r => r.id)).toEqual(['sess-live-a'])

    // The first prompt persisted the row and the recents page came back with it
    // while the live snapshot is still the previous poll's — the stored refresh
    // is trailing-throttled (SESSIONS_LIST_TICK_GAP_MS) and a typing burst
    // defers it further, so write-time dedupe alone would render both.
    setSessions([makeSessionInfo({ id: 'sess-live-a', message_count: 1 })])

    expect($visibleLiveSessions.get()).toEqual([])
  })

  it('keeps the array identity when nothing is filtered', () => {
    reconcileLiveSessions({ sessions: [liveItem()] }, OPTS)

    expect($visibleLiveSessions.get()).toBe($liveSessions.get())
  })
})
