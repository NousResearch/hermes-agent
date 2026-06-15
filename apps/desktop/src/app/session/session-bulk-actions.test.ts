import { beforeEach, describe, expect, it, vi } from 'vitest'

import { deleteSession, setSessionArchived } from '@/hermes'
import { $pinnedSessionIds } from '@/store/layout'
import {
  $archivedSessions,
  $archivedSessionsTotal,
  $cronSessions,
  $messagingPlatformTotals,
  $messagingSessions,
  $sessionProfileTotals,
  $sessions,
  $sessionsTotal
} from '@/store/session'
import type { SessionInfo } from '@/types/hermes'

import {
  archiveStoredSessions,
  deleteStoredSessions,
  hideStoredSession,
  restoreArchivedSessions
} from './session-bulk-actions'

vi.mock(import('@/hermes'), async importOriginal => {
  const actual = await importOriginal()

  return {
    ...actual,
    deleteSession: vi.fn().mockResolvedValue({ ok: true }),
    setSessionArchived: vi.fn().mockResolvedValue({ ok: true })
  }
})

const mockedSetArchived = vi.mocked(setSessionArchived)
const mockedDelete = vi.mocked(deleteSession)

function session(over: Partial<SessionInfo> & { id: string }): SessionInfo {
  return {
    archived: false,
    cwd: null,
    ended_at: null,
    _lineage_root_id: null,
    input_tokens: 0,
    is_active: false,
    last_active: 1000,
    message_count: 3,
    model: null,
    output_tokens: 0,
    preview: null,
    profile: 'default',
    source: null,
    started_at: 1000,
    title: null,
    tool_call_count: 0,
    ...over
  } as SessionInfo
}

beforeEach(() => {
  vi.clearAllMocks()
  mockedSetArchived.mockResolvedValue({ ok: true })
  mockedDelete.mockResolvedValue({ ok: true })

  $sessions.set([session({ id: 'a1' }), session({ id: 'a2', profile: 'work' }), session({ id: 'a3' })])
  $sessionsTotal.set(10)
  $sessionProfileTotals.set({ default: 7, work: 3 })
  $messagingSessions.set([session({ id: 'm1', source: 'telegram' })])
  $messagingPlatformTotals.set({ telegram: 5 })
  $cronSessions.set([])
  $archivedSessions.set([session({ archived: true, id: 'z1' })])
  $archivedSessionsTotal.set(4)
  $pinnedSessionIds.set([])
})

describe('archiveStoredSessions', () => {
  it('archives agent rows: slices, agent+profile totals, and the Archived slice all stay honest', async () => {
    const result = await archiveStoredSessions(['a1', 'a2'], { silent: true })

    expect(result.ok.map(s => s.id).sort()).toEqual(['a1', 'a2'])
    expect($sessions.get().map(s => s.id)).toEqual(['a3'])
    expect($sessionsTotal.get()).toBe(8)
    // Per-profile totals (the scoped Load-more math) decrement per owning profile.
    expect($sessionProfileTotals.get()).toEqual({ default: 6, work: 2 })
    // Rows surface in the Archived section immediately.
    expect($archivedSessions.get().map(s => s.id)).toContain('a1')
    expect($archivedSessionsTotal.get()).toBe(6)
    expect(mockedSetArchived).toHaveBeenCalledWith('a1', true, 'default')
    expect(mockedSetArchived).toHaveBeenCalledWith('a2', true, 'work')
  })

  it('archives a messaging row against the PLATFORM totals, never the agent totals', async () => {
    await archiveStoredSessions(['m1'], { silent: true })

    // The original phantom-"Load 1 more" bug: agent totals must not move.
    expect($sessionsTotal.get()).toBe(10)
    expect($sessionProfileTotals.get()).toEqual({ default: 7, work: 3 })
    expect($messagingSessions.get()).toEqual([])
    expect($messagingPlatformTotals.get()).toEqual({ telegram: 4 })
  })

  it('rolls back only the failed rows and reports both outcomes', async () => {
    mockedSetArchived.mockImplementation(id =>
      id === 'a1' ? Promise.reject(new Error('boom')) : Promise.resolve({ ok: true })
    )

    const result = await archiveStoredSessions(['a1', 'a2'], { silent: true })

    expect(result.ok.map(s => s.id)).toEqual(['a2'])
    expect(result.failed.map(s => s.id)).toEqual(['a1'])
    // a1 came back (front of list), a2 stayed archived.
    expect($sessions.get().map(s => s.id)).toEqual(['a1', 'a3'])
    expect($sessionsTotal.get()).toBe(9)
    expect($sessionProfileTotals.get()).toEqual({ default: 7, work: 2 })
    expect($archivedSessions.get().some(s => s.id === 'a1')).toBe(false)
    expect($archivedSessions.get().some(s => s.id === 'a2')).toBe(true)
  })

  it('strips pins (live + lineage-root ids) and restores them on failure', async () => {
    $sessions.set([session({ id: 'tip', _lineage_root_id: 'root' })])
    $pinnedSessionIds.set(['root', 'other'])
    mockedSetArchived.mockRejectedValue(new Error('offline'))

    await archiveStoredSessions(['tip'], { silent: true })

    expect($pinnedSessionIds.get()).toContain('root')
    expect($pinnedSessionIds.get()).toContain('other')

    mockedSetArchived.mockResolvedValue({ ok: true })
    await archiveStoredSessions(['tip'], { silent: true })
    expect($pinnedSessionIds.get()).toEqual(['other'])
  })

  it('skips rows already archived', async () => {
    const result = await archiveStoredSessions(['z1'], { silent: true })

    expect(result.ok).toEqual([])
    expect(mockedSetArchived).not.toHaveBeenCalled()
  })
})

describe('restoreArchivedSessions', () => {
  it('moves an archived row back to recents with totals incremented', async () => {
    const result = await restoreArchivedSessions(['z1'], { silent: true })

    expect(result.ok.map(s => s.id)).toEqual(['z1'])
    expect($archivedSessions.get()).toEqual([])
    expect($archivedSessionsTotal.get()).toBe(3)
    expect($sessions.get().some(s => s.id === 'z1' && s.archived === false)).toBe(true)
    expect($sessionsTotal.get()).toBe(11)
    expect($sessionProfileTotals.get()).toEqual({ default: 8, work: 3 })
    expect(mockedSetArchived).toHaveBeenCalledWith('z1', false, 'default')
  })

  it('restores a messaging-source row into the messaging slice', async () => {
    $archivedSessions.set([session({ archived: true, id: 'zt', source: 'telegram' })])

    await restoreArchivedSessions(['zt'], { silent: true })

    expect($messagingSessions.get().some(s => s.id === 'zt')).toBe(true)
    expect($messagingPlatformTotals.get()).toEqual({ telegram: 6 })
    // Agent totals untouched.
    expect($sessionsTotal.get()).toBe(10)
  })

  it('puts the row back in Archived when the PATCH fails', async () => {
    mockedSetArchived.mockRejectedValue(new Error('offline'))

    const result = await restoreArchivedSessions(['z1'], { silent: true })

    expect(result.failed.map(s => s.id)).toEqual(['z1'])
    expect($archivedSessions.get().map(s => s.id)).toEqual(['z1'])
    expect($archivedSessionsTotal.get()).toBe(4)
    expect($sessions.get().some(s => s.id === 'z1')).toBe(false)
    expect($sessionsTotal.get()).toBe(10)
  })

  it('only acts on rows currently in the Archived slice', async () => {
    const result = await restoreArchivedSessions(['a1'], { silent: true })

    expect(result.ok).toEqual([])
    expect(mockedSetArchived).not.toHaveBeenCalled()
  })
})

describe('deleteStoredSessions', () => {
  it('deletes across slices, including the Archived section', async () => {
    const result = await deleteStoredSessions(['a1', 'z1'], { silent: true })

    expect(result.ok.map(s => s.id).sort()).toEqual(['a1', 'z1'])
    expect($sessions.get().some(s => s.id === 'a1')).toBe(false)
    expect($sessionsTotal.get()).toBe(9)
    expect($archivedSessions.get()).toEqual([])
    expect($archivedSessionsTotal.get()).toBe(3)
  })

  it('treats "session not found" as success', async () => {
    mockedDelete.mockRejectedValue(new Error('Session not found'))

    const result = await deleteStoredSessions(['a1'], { silent: true })

    expect(result.ok.map(s => s.id)).toEqual(['a1'])
    expect($sessions.get().some(s => s.id === 'a1')).toBe(false)
  })

  it('rolls the row back into its slice on a real failure', async () => {
    mockedDelete.mockRejectedValue(new Error('locked'))

    const result = await deleteStoredSessions(['m1'], { silent: true })

    expect(result.failed.map(s => s.id)).toEqual(['m1'])
    expect($messagingSessions.get().map(s => s.id)).toEqual(['m1'])
    expect($messagingPlatformTotals.get()).toEqual({ telegram: 5 })
  })
})

describe('hideStoredSession', () => {
  it('hides + undoes an agent row with profile totals intact', () => {
    const hidden = hideStoredSession('a2')

    expect(hidden?.slice).toBe('agent')
    expect($sessions.get().some(s => s.id === 'a2')).toBe(false)
    expect($sessionsTotal.get()).toBe(9)
    expect($sessionProfileTotals.get()).toEqual({ default: 7, work: 2 })

    hidden?.undo()
    expect($sessions.get().some(s => s.id === 'a2')).toBe(true)
    expect($sessionsTotal.get()).toBe(10)
    expect($sessionProfileTotals.get()).toEqual({ default: 7, work: 3 })
  })

  it('hides a messaging row without touching agent totals', () => {
    const hidden = hideStoredSession('m1')

    expect(hidden?.slice).toBe('messaging')
    expect($sessionsTotal.get()).toBe(10)
    expect($messagingPlatformTotals.get()).toEqual({ telegram: 4 })

    hidden?.undo()
    expect($messagingPlatformTotals.get()).toEqual({ telegram: 5 })
  })

  it('returns null for unknown ids', () => {
    expect(hideStoredSession('nope')).toBeNull()
  })
})
