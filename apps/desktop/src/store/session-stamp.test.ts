import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { SessionInfo } from '@/types/hermes'

const patch = vi.fn<(id: string, stamp: null | string, profile?: null | string) => Promise<{ ok: boolean }>>(() =>
  Promise.resolve({ ok: true })
)

vi.mock('@/hermes', () => ({
  // The session store reaches the profile store, which sets the request profile
  // at import time; this suite only cares about the stamp call.
  setApiRequestProfile: () => {},
  setSessionStampRemote: (id: string, stamp: null | string, profile?: null | string) => patch(id, stamp, profile)
}))

import { $cronSessions, $messagingSessions, $sessions } from '@/store/session'
import { $archivedSessions } from '@/store/sidebar-archive'

import { $sessionStamps, applySessionStamp, normalizeSessionStamp } from './session-stamp'

const row = (id: string, extra: Partial<SessionInfo> = {}): SessionInfo =>
  ({ id, message_count: 1, source: 'cli', started_at: 0, title: id, ...extra }) as SessionInfo

beforeEach(() => {
  $sessions.set([])
  $cronSessions.set([])
  $messagingSessions.set([])
  $archivedSessions.set([])
  patch.mockClear()
})

afterEach(() => {
  $sessions.set([])
  $cronSessions.set([])
  $messagingSessions.set([])
  $archivedSessions.set([])
})

describe('normalizeSessionStamp', () => {
  it('trims, collapses interior whitespace and caps the length', () => {
    expect(normalizeSessionStamp('  Merged  ')).toBe('Merged')
    expect(normalizeSessionStamp('Waiting   on\n CI')).toBe('Waiting on CI')
    expect(normalizeSessionStamp('x'.repeat(40))?.length).toBe(24)
  })

  it('treats empty, whitespace-only and null as no stamp', () => {
    expect(normalizeSessionStamp('')).toBeNull()
    expect(normalizeSessionStamp('   ')).toBeNull()
    expect(normalizeSessionStamp(null)).toBeNull()
    expect(normalizeSessionStamp(undefined)).toBeNull()
  })
})

describe('applySessionStamp', () => {
  it('paints the stamp before the backend answers, and persists the normalized label', async () => {
    $sessions.set([row('a', { profile: 'work' })])

    const pending = applySessionStamp('a', 'work', '  wip  ')

    // Optimistic: the row already shows it, so a slow round trip never reads as
    // "the click did nothing".
    expect($sessions.get()[0].stamp).toBe('wip')

    await pending

    expect(patch).toHaveBeenCalledWith('a', 'wip', 'work')
    expect($sessions.get()[0].stamp).toBe('wip')
  })

  it('clears the stamp when handed an empty label', async () => {
    $sessions.set([row('a', { stamp: 'Merged' })])

    await applySessionStamp('a', undefined, '')

    expect(patch).toHaveBeenCalledWith('a', null, undefined)
    expect($sessions.get()[0].stamp).toBeNull()
  })

  it('puts the row back when the backend refuses, rather than leaving a stamp that is not there', async () => {
    patch.mockRejectedValueOnce(new Error('offline'))
    $sessions.set([row('a', { stamp: 'Hold' })])

    const ok = await applySessionStamp('a', undefined, 'Merged')

    expect(ok).toBe(false)
    expect($sessions.get()[0].stamp).toBe('Hold')
  })

  it('patches every list that can hold the row, and leaves other rows untouched', async () => {
    $archivedSessions.set([row('a', { profile: 'default' })])
    $sessions.set([row('b')])

    await applySessionStamp('a', 'default', 'Review')

    expect($archivedSessions.get()[0].stamp).toBe('Review')
    expect($sessions.get()[0].stamp).toBeUndefined()
  })
})

describe('$sessionStamps', () => {
  it('maps live and lineage ids to the stamp, and skips unstamped rows', () => {
    $sessions.set([
      row('tip', { _lineage_ids: ['mid', 'root'], _lineage_root_id: 'root', stamp: 'Handoff' }),
      row('plain')
    ])

    expect($sessionStamps.get().get('tip')).toBe('Handoff')
    expect($sessionStamps.get().get('root')).toBe('Handoff')
    expect($sessionStamps.get().get('mid')).toBe('Handoff')
    expect($sessionStamps.get().has('plain')).toBe(false)
  })
})
