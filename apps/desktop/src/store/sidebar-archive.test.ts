import { beforeEach, describe, expect, it, vi } from 'vitest'

import { listAllProfileSessions } from '@/hermes'

import { $archivedSessions, $archivedSessionsLoading, loadArchivedSessions } from './sidebar-archive'

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal()),
  listAllProfileSessions: vi.fn()
}))

const archivedRows = (ids: string[]) => ids.map(id => ({ id }) as never)

function mockArchivedFetch(ids: string[]): void {
  vi.mocked(listAllProfileSessions).mockResolvedValueOnce({
    sessions: archivedRows(ids),
    total: ids.length
  } as never)
}

beforeEach(() => {
  $archivedSessions.set([])
  $archivedSessionsLoading.set(false)
  vi.mocked(listAllProfileSessions).mockReset()
  // Default: an empty successful page, so a test that forgets to arm a mock
  // still exercises the success path rather than silently failing.
  vi.mocked(listAllProfileSessions).mockResolvedValue({ sessions: [], total: 0 } as never)
})

describe('loadArchivedSessions', () => {
  it('fetches the archived-only page and replaces the set on success', async () => {
    mockArchivedFetch(['first', 'second'])
    await loadArchivedSessions()
    expect(listAllProfileSessions).toHaveBeenCalledWith(200, 0, 'only')
    expect($archivedSessions.get().map(session => session.id)).toEqual(['first', 'second'])

    mockArchivedFetch(['third'])
    await loadArchivedSessions()
    expect($archivedSessions.get().map(session => session.id)).toEqual(['third'])
  })

  it('retains previously loaded rows when a reload fails (#111397)', async () => {
    mockArchivedFetch(['keep-a', 'keep-b'])
    await loadArchivedSessions()
    expect($archivedSessions.get()).toHaveLength(2)

    vi.mocked(listAllProfileSessions).mockRejectedValueOnce(new Error('gateway down'))
    await loadArchivedSessions()

    // A refresh is new information layered over what we know — a transient
    // backend error must not blank an open Archived view.
    expect($archivedSessions.get().map(session => session.id)).toEqual(['keep-a', 'keep-b'])
  })

  it('releases the loading guard after a failed fetch so later reloads proceed', async () => {
    vi.mocked(listAllProfileSessions).mockRejectedValueOnce(new Error('gateway down'))
    await loadArchivedSessions()
    expect($archivedSessionsLoading.get()).toBe(false)

    mockArchivedFetch(['after-failure'])
    await loadArchivedSessions()
    expect($archivedSessions.get().map(session => session.id)).toEqual(['after-failure'])
    expect($archivedSessionsLoading.get()).toBe(false)
  })

  it('stays empty when the very first load fails', async () => {
    vi.mocked(listAllProfileSessions).mockRejectedValueOnce(new Error('gateway down'))
    await loadArchivedSessions()
    expect($archivedSessions.get()).toEqual([])
  })
})
