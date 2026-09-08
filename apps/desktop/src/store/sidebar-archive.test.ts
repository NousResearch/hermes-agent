import { afterEach, describe, expect, it, vi } from 'vitest'

import { listAllProfileSessions, type SessionInfo } from '@/hermes'

import { $archivedSessions, $archivedSessionsLoading, loadArchivedSessions } from './sidebar-archive'

vi.mock('@/hermes', () => ({ listAllProfileSessions: vi.fn() }))

const row = (id: string) => ({ id }) as SessionInfo

afterEach(() => {
  $archivedSessions.set([])
  $archivedSessionsLoading.set(false)
  vi.clearAllMocks()
})

describe('loadArchivedSessions', () => {
  it('lets a changed scope supersede an in-flight load', async () => {
    let resolveCurrent!: (value: { sessions: SessionInfo[] }) => void

    const current = new Promise<{ sessions: SessionInfo[] }>(resolve => {
      resolveCurrent = resolve
    })

    vi.mocked(listAllProfileSessions)
      .mockReturnValueOnce(current as ReturnType<typeof listAllProfileSessions>)
      .mockResolvedValueOnce({ sessions: [row('all-gateways')] } as never)

    const staleLoad = loadArchivedSessions(false)
    await loadArchivedSessions(true)
    resolveCurrent({ sessions: [row('current-gateway')] })
    await staleLoad

    expect(listAllProfileSessions).toHaveBeenCalledTimes(2)
    expect($archivedSessions.get().map(session => session.id)).toEqual(['all-gateways'])
    expect($archivedSessionsLoading.get()).toBe(false)
  })
})
