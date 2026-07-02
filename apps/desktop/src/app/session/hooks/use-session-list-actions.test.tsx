import { cleanup, render, waitFor } from '@testing-library/react'
import { useEffect } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { bulkDeleteSessions, listAllProfileSessions, type SessionInfo } from '@/hermes'
import { $pinnedSessionIds } from '@/store/layout'
import { $sessions, setSessions, setSessionsTotal } from '@/store/session'

import { useSessionListActions } from './use-session-list-actions'

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  bulkDeleteSessions: vi.fn(),
  getCronJobs: vi.fn().mockResolvedValue([]),
  listAllProfileSessions: vi.fn()
}))

const mockedList = vi.mocked(listAllProfileSessions)
const mockedBulkDelete = vi.mocked(bulkDeleteSessions)

function storedSession(overrides: Partial<SessionInfo> = {}): SessionInfo {
  return {
    ended_at: null,
    id: 'stored-1',
    input_tokens: 0,
    is_active: false,
    last_active: 1,
    message_count: 1,
    model: null,
    output_tokens: 0,
    preview: null,
    profile: 'default',
    source: 'desktop',
    started_at: 1,
    title: 'stored',
    tool_call_count: 0,
    ...overrides
  }
}

function emptyPage() {
  return { limit: 0, offset: 0, sessions: [] as SessionInfo[], total: 0 }
}

function Harness({ onReady }: { onReady: (clear: () => Promise<number>) => void }) {
  const { clearAllSessions } = useSessionListActions({ profileScope: 'default' })

  useEffect(() => {
    onReady(clearAllSessions)
  }, [clearAllSessions, onReady])

  return null
}

async function getClear(): Promise<() => Promise<number>> {
  let clear: (() => Promise<number>) | null = null
  render(<Harness onReady={c => (clear = c)} />)
  await waitFor(() => expect(clear).not.toBeNull())

  return clear!
}

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
  setSessions([])
  setSessionsTotal(0)
  $pinnedSessionIds.set([])
})

describe('useSessionListActions › clearAllSessions', () => {
  it('pages the scope, bulk-deletes every chat, and clears the list + pins', async () => {
    const rows = [storedSession({ id: 's1' }), storedSession({ id: 's2' })]
    setSessions(rows)
    setSessionsTotal(2)
    $pinnedSessionIds.set(['s2'])

    let drained = false
    mockedList.mockImplementation((limit, _min, _archived, _order, _profile, filter) => {
      // Cron / messaging slices fetched by the closing refresh stay empty.
      if (filter?.source) {
        return Promise.resolve(emptyPage())
      }

      // The clear loop pages with limit === BULK_DELETE_MAX_IDS (500): hand back
      // the rows once, then empty so the loop terminates.
      if (limit === 500 && !drained) {
        drained = true

        return Promise.resolve({ limit: 500, offset: 0, sessions: rows, total: 2 })
      }

      return Promise.resolve(emptyPage())
    })
    mockedBulkDelete.mockImplementation((ids: string[]) => Promise.resolve({ deleted: ids.length, ok: true }))

    const clear = await getClear()
    const removed = await clear()

    expect(removed).toBe(2)
    expect(mockedBulkDelete).toHaveBeenCalledTimes(1)
    expect(mockedBulkDelete).toHaveBeenCalledWith(['s1', 's2'], 'default')
    await waitFor(() => expect($sessions.get()).toHaveLength(0))
    expect($pinnedSessionIds.get()).toEqual([])
  })

  it('groups ids by owning profile so each profile is deleted against its own db', async () => {
    const rows = [
      storedSession({ id: 'a1', profile: 'default' }),
      storedSession({ id: 'b1', profile: 'work' }),
      storedSession({ id: 'a2', profile: 'default' })
    ]

    let drained = false
    mockedList.mockImplementation((limit, _min, _archived, _order, _profile, filter) => {
      if (filter?.source) {
        return Promise.resolve(emptyPage())
      }

      if (limit === 500 && !drained) {
        drained = true

        return Promise.resolve({ limit: 500, offset: 0, sessions: rows, total: rows.length })
      }

      return Promise.resolve(emptyPage())
    })
    mockedBulkDelete.mockImplementation((ids: string[]) => Promise.resolve({ deleted: ids.length, ok: true }))

    const clear = await getClear()
    await clear()

    expect(mockedBulkDelete).toHaveBeenCalledWith(['a1', 'a2'], 'default')
    expect(mockedBulkDelete).toHaveBeenCalledWith(['b1'], 'work')
  })

  it('is a no-op (no delete calls) when the scope is already empty', async () => {
    mockedList.mockResolvedValue(emptyPage())

    const clear = await getClear()
    const removed = await clear()

    expect(removed).toBe(0)
    expect(mockedBulkDelete).not.toHaveBeenCalled()
  })
})
