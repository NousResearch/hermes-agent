import { act, renderHook } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import type { SessionInfo, SidebarSessionsResponse } from '@/hermes'
import { ensureGatewayForAgent, setPrimaryGateway, setPrimaryGatewayConnection } from '@/store/gateway'
import { $sessions, setSessions } from '@/store/session'

import { deferred } from '../../../test/deferred'

import { useSessionListActions } from './use-session-list-actions'

// Real gateway registry, stubbed transport: the activation epoch here is the
// one production bumps, not a test double.
const listSidebarSessions = vi.fn()

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  getCronJobs: vi.fn(async () => []),
  listSidebarSessions: (...args: unknown[]) => listSidebarSessions(...args)
}))

const row = (id: string): SessionInfo =>
  ({
    id,
    last_active: 1000,
    message_count: 3,
    profile: 'default',
    source: 'desktop',
    started_at: 900,
    title: `Chat ${id}`
  }) as SessionInfo

const page = (sessions: SessionInfo[]): SidebarSessionsResponse => ({
  recents: { sessions },
  cron: { sessions: [] },
  messaging: { sessions: [] }
})

afterEach(() => setSessions([]))

// #67600 / #88866: resuming a default-profile chat on the local source routes
// through ensureGatewayAgent('local', 'default'). That activation lands on the
// socket the window already shows, so no route atom moves and no effect asks
// for the list again; the sidebar page in flight at that moment is the only
// one the window gets.
it('keeps the default-profile sidebar when the active route is re-activated mid-refresh', async () => {
  const primary = { connectionState: 'open', request: vi.fn(async () => ({})) }
  setPrimaryGateway(primary as never, 'default')
  setPrimaryGatewayConnection({ connectionId: 'local', mode: 'local' } as never)

  const pending = deferred<SidebarSessionsResponse>()
  const rows = [row('a'), row('b')]
  listSidebarSessions.mockReturnValueOnce(pending.promise).mockResolvedValue(page(rows))

  const { result } = renderHook(() => useSessionListActions({ profileScope: 'default' }))
  let refresh!: Promise<void>

  act(() => {
    refresh = result.current.refreshSessions()
  })

  await ensureGatewayForAgent('local', 'default')

  await act(async () => {
    pending.resolve(page(rows))
    await refresh
  })

  expect($sessions.get().map(session => session.id)).toEqual(['a', 'b'])
})
