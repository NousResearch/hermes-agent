import { beforeEach, describe, expect, it, vi } from 'vitest'

import { $apiRequestScope } from '@/api/client'
import type { SessionInfo } from '@/types/hermes'

import { $sidebarShowArchived } from './layout'
import {
  $archivedSessions,
  $archivedSessionsLoading,
  listEveryArchivedSession,
  loadArchivedSessions
} from './sidebar-archive'

const listAllProfileSessions = vi.hoisted(() => vi.fn())

vi.mock('@/api/sessions', () => ({
  listAllProfileSessions
}))

const PAGE_SIZE = 200

const row = (id: string, overrides: Partial<SessionInfo> = {}): SessionInfo =>
  ({
    id,
    archived: true,
    input_tokens: 0,
    output_tokens: 0,
    message_count: 1,
    started_at: 1,
    last_active: 1,
    source: 'desktop',
    tool_call_count: 0,
    ...overrides
  }) as SessionInfo

/** A back-filled pinned row: appended past the window by the server, ignores
 * the archived filter, and repeats on every page whose window covers it. */
const pinnedRow = (id: string): SessionInfo => row(id, { archived: false, pinned: true })

/** `id`-numbered archived rows for a full page of the given size. */
const page = (from: number, count: number): SessionInfo[] =>
  Array.from({ length: count }, (_, i) => row(`a-${from + i}`))

beforeEach(() => {
  $sidebarShowArchived.set(false)
  $apiRequestScope.set({ connectionId: null, profile: null })
  $archivedSessions.set([])
  listAllProfileSessions.mockReset()
})

describe('loadArchivedSessions', () => {
  it('keeps same-id archived sessions from different profiles across pages', async () => {
    listAllProfileSessions
      .mockResolvedValueOnce({
        sessions: [row('same', { profile: 'default' }), ...page(1, PAGE_SIZE - 1)],
        total: PAGE_SIZE + 1
      })
      .mockResolvedValueOnce({ sessions: [row('same', { profile: 'work' })], total: PAGE_SIZE + 1 })

    const sessions = await listEveryArchivedSession()

    expect(sessions.filter(session => session.id === 'same').map(session => session.profile)).toEqual([
      'default',
      'work'
    ])
  })

  it('keeps the last successful result when a refresh fails', async () => {
    const existing = { id: 'archived-1', title: 'Keep me' } as SessionInfo
    $archivedSessions.set([existing])
    listAllProfileSessions.mockRejectedValue(new Error('offline'))

    await loadArchivedSessions()

    expect($archivedSessions.get()).toEqual([existing])
  })

  it('does not publish the previous connection rows after a switch mid-load', async () => {
    let resolvePrevious!: (page: unknown) => void
    let resolveCurrent!: (page: unknown) => void

    listAllProfileSessions
      .mockImplementationOnce(() => new Promise(resolve => (resolvePrevious = resolve)))
      .mockImplementationOnce(() => new Promise(resolve => (resolveCurrent = resolve)))

    $apiRequestScope.setKey('connectionId', 'conn-a')
    const previous = loadArchivedSessions()
    $apiRequestScope.setKey('connectionId', 'conn-b')
    const current = loadArchivedSessions()

    // The new connection's reload must fetch, not be swallowed by A's flag.
    expect(listAllProfileSessions).toHaveBeenCalledTimes(2)

    resolveCurrent({ sessions: [row('b-archived')], total: 1 })
    await current
    resolvePrevious({ sessions: [row('a-archived')], total: 1 })
    await previous

    expect($archivedSessions.get().map(session => session.id)).toEqual(['b-archived'])
    expect($archivedSessionsLoading.get()).toBe(false)
  })

  it('clears the previous connection rows when a load starts on a new one', async () => {
    let resolveCurrent!: (page: unknown) => void

    listAllProfileSessions
      .mockResolvedValueOnce({ sessions: [row('a-archived')], total: 1 })
      .mockImplementationOnce(() => new Promise(resolve => (resolveCurrent = resolve)))

    $apiRequestScope.setKey('connectionId', 'conn-a')
    await loadArchivedSessions()
    $apiRequestScope.setKey('connectionId', 'conn-b')
    const current = loadArchivedSessions()

    expect($archivedSessions.get()).toEqual([])

    resolveCurrent({ sessions: [row('b-archived')], total: 1 })
    await current

    expect($archivedSessions.get().map(session => session.id)).toEqual(['b-archived'])
  })

  // Nothing else reloads on a scope switch: the sidebar effect watches only the Archived toggle
  // and background sync waits for a backend sessions.changed event.
  it('clears and reloads an open Archived view on a scope switch with no other trigger', async () => {
    $sidebarShowArchived.set(true)
    let resolveCurrent!: (page: unknown) => void

    listAllProfileSessions
      .mockResolvedValueOnce({ sessions: [row('a-archived')], total: 1 })
      .mockImplementationOnce(() => new Promise(resolve => (resolveCurrent = resolve)))

    await loadArchivedSessions()
    $apiRequestScope.setKey('profile', 'work')

    expect($archivedSessions.get()).toEqual([])
    expect(listAllProfileSessions).toHaveBeenCalledTimes(2)

    resolveCurrent({ sessions: [row('work-archived')], total: 1 })
    await vi.waitFor(() => expect($archivedSessions.get().map(session => session.id)).toEqual(['work-archived']))
    expect($archivedSessionsLoading.get()).toBe(false)
  })

  it('replaces a first load stranded by a scope switch while Archived is open', async () => {
    $sidebarShowArchived.set(true)
    let resolvePrevious!: (page: unknown) => void

    listAllProfileSessions
      .mockImplementationOnce(() => new Promise(resolve => (resolvePrevious = resolve)))
      .mockResolvedValueOnce({ sessions: [row('b-archived')], total: 1 })

    const previous = loadArchivedSessions()
    $apiRequestScope.setKey('connectionId', 'conn-b')

    await vi.waitFor(() => expect($archivedSessions.get().map(session => session.id)).toEqual(['b-archived']))
    resolvePrevious({ sessions: [row('a-archived')], total: 1 })
    await previous

    expect($archivedSessions.get().map(session => session.id)).toEqual(['b-archived'])
    expect($archivedSessionsLoading.get()).toBe(false)
  })

  it('clears a closed Archived view on a scope switch without fetching', async () => {
    $archivedSessions.set([row('a-archived')])

    $apiRequestScope.setKey('profile', 'work')

    expect($archivedSessions.get()).toEqual([])
    expect(listAllProfileSessions).not.toHaveBeenCalled()
  })
})

describe('listEveryArchivedSession', () => {
  it('keeps paging when a live pin displaced the last archived row', async () => {
    listAllProfileSessions
      .mockResolvedValueOnce({ sessions: [pinnedRow('pin-x'), ...page(0, PAGE_SIZE - 1)], total: PAGE_SIZE })
      .mockResolvedValueOnce({ sessions: [row(`a-${PAGE_SIZE - 1}`), pinnedRow('pin-x')], total: PAGE_SIZE })

    const sessions = await listEveryArchivedSession()

    expect(sessions.filter(session => session.archived)).toHaveLength(PAGE_SIZE)
    expect(sessions.filter(session => session.id === 'pin-x')).toHaveLength(1)
    expect(listAllProfileSessions).toHaveBeenCalledTimes(2)
  })

  it('stops past the total when only live pins arrive', async () => {
    // A row archived mid-load behind the cursor leaves the count one short,
    // and the backend still back-fills the live pin past the end.
    listAllProfileSessions.mockImplementation(async (...args: unknown[]) => {
      const offset = args[6] as number

      if (offset > 4 * PAGE_SIZE) {
        throw new Error('archived paging did not stop')
      }

      return { sessions: [...page(offset, Math.max(0, PAGE_SIZE - offset)), pinnedRow('pin-x')], total: PAGE_SIZE + 1 }
    })

    const sessions = await listEveryArchivedSession()

    expect(sessions.filter(session => session.archived)).toHaveLength(PAGE_SIZE)
    expect(listAllProfileSessions).toHaveBeenCalledTimes(2)
  })

  it('paginates until the backend total is loaded', async () => {
    const firstPage = Array.from({ length: 200 }, (_, i) => row(`a${i}`))

    listAllProfileSessions
      .mockResolvedValueOnce({ sessions: firstPage, total: 201, limit: 200, offset: 0 })
      .mockResolvedValueOnce({ sessions: [row('c')], total: 201, limit: 200, offset: 200 })

    await expect(listEveryArchivedSession()).resolves.toEqual([...firstPage, row('c')])
    expect(listAllProfileSessions).toHaveBeenNthCalledWith(1, 200, 0, 'only', 'recent', 'all', {}, 0)
    expect(listAllProfileSessions).toHaveBeenNthCalledWith(2, 200, 0, 'only', 'recent', 'all', {}, 200)
  })

  it('stops on an empty page when profiles changed during pagination', async () => {
    const firstPage = Array.from({ length: 200 }, (_, i) => row(`a${i}`))

    listAllProfileSessions
      .mockResolvedValueOnce({ sessions: firstPage, total: 400, limit: 200, offset: 0 })
      .mockResolvedValueOnce({ sessions: [], total: 400, limit: 200, offset: 200 })

    await expect(listEveryArchivedSession()).resolves.toEqual(firstPage)
    expect(listAllProfileSessions).toHaveBeenCalledTimes(2)
  })

  it('advances the offset by the page size, not by the row count', async () => {
    // Server contract: a page returns limit window rows PLUS every back-filled
    // pinned row, so pages 1 and 2 here both hand back 200 archived rows and
    // a riding pin. The next offset must stay 200/400 regardless.
    listAllProfileSessions
      .mockResolvedValueOnce({
        sessions: [...page(0, PAGE_SIZE), pinnedRow('pin-x')],
        total: 401,
        limit: PAGE_SIZE,
        offset: 0
      })
      .mockResolvedValueOnce({
        sessions: [...page(PAGE_SIZE, PAGE_SIZE), pinnedRow('pin-x')],
        total: 401,
        limit: PAGE_SIZE,
        offset: PAGE_SIZE
      })
      .mockResolvedValueOnce({ sessions: [row('a-400')], total: 401, limit: PAGE_SIZE, offset: 2 * PAGE_SIZE })

    const sessions = await listEveryArchivedSession()

    // All 401 archived rows plus the pin — none skipped, pin listed once.
    expect(sessions).toHaveLength(402)
    expect(sessions.filter(s => s.id === 'pin-x')).toHaveLength(1)
    expect(sessions.filter(s => s.archived).map(s => s.id)).toContain('a-400')
    expect(listAllProfileSessions).toHaveBeenNthCalledWith(2, PAGE_SIZE, 0, 'only', 'recent', 'all', {}, PAGE_SIZE)
    expect(listAllProfileSessions).toHaveBeenNthCalledWith(3, PAGE_SIZE, 0, 'only', 'recent', 'all', {}, 2 * PAGE_SIZE)
  })

  it('loads every archived row when non-archived pins inflate each page', async () => {
    // Reviewer's worst case: 1000 archived rows; pins live outside the archive
    // and total counts only archived rows. pages 1-5 are full 200-row windows
    // each carrying 100 extra pin rows, so the row count crosses 1000 on page
    // 5 — the loop must still stop on the cursor, not on the inflated count.
    const total = 1000
    let calls = 0
    listAllProfileSessions.mockImplementation(async (...args: unknown[]) => {
      const offset = args[6] as number
      calls += 1
      const remaining = Math.max(0, Math.min(PAGE_SIZE, total - offset))
      const pins = Array.from({ length: 100 }, (_, i) => pinnedRow(`pin-${i}`))

      return {
        sessions: [...page(offset, remaining), ...pins],
        total,
        limit: PAGE_SIZE,
        offset
      }
    })

    const sessions = await listEveryArchivedSession()

    // 5 pages: offset 0,200,400,600,800 — the 6th request is never issued
    // because offset 1000 has reached total.
    expect(calls).toBe(5)
    expect(sessions.filter(s => s.archived)).toHaveLength(total)
    // The 100 distinct pins ride along once each, not once per page.
    expect(sessions.filter(s => !s.archived)).toHaveLength(100)
  })
})
