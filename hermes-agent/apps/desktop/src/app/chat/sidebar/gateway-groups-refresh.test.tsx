// @vitest-environment jsdom
import { act, cleanup, render, renderHook, within } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { setApiRequestConnection } from '@/api/client'
import { useSessionListActions } from '@/app/session/hooks/use-session-list-actions'
import { SidebarProvider } from '@/components/ui/sidebar'
import type { SessionInfo, SidebarSessionsResponse } from '@/hermes'
import { $connectionsRegistry } from '@/store/connection-registry-state'
import { setSidebarGrouping } from '@/store/layout'
import { $profiles, $showAllProfiles } from '@/store/profile'
import { $sessions } from '@/store/session'
import { makeSessionInfo } from '@/test/session-info'

import { ChatSidebar } from './index'

// End to end from the sidebar refresh into the grouped sidebar: only the API response is stubbed,
// the real refresh writes the real stores and the real ChatSidebar groups them.

const listSidebarSessions = vi.fn()

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  getCronJobs: async () => [],
  listSidebarSessions: (...args: unknown[]) => listSidebarSessions(...args)
}))

const noop = () => {}

const noopAsync = async () => {}

const chat = (id: string, profile: string): SessionInfo =>
  makeSessionInfo({ id, profile, title: `${id} title`, last_active: Date.now() / 1000 })

const page = (
  sessions: SessionInfo[],
  errors?: Array<{ error: string; profile: string }>
): SidebarSessionsResponse => ({
  recents: { sessions, ...(errors ? { errors } : {}) },
  cron: { sessions: [] },
  messaging: { sessions: [] }
})

function mountSidebar() {
  render(
    <MemoryRouter>
      <SidebarProvider>
        <ChatSidebar
          currentView="chat"
          onArchiveSession={noop}
          onBranchSession={noop}
          onDeleteSession={noop}
          onLoadMoreSessions={noop}
          onManageCronJob={noop}
          onNavigate={noop}
          onNewSessionInWorkspace={noop}
          onNewSessionSplit={noop}
          onResumeSession={noop}
          onRetrySessions={noopAsync}
          onTriggerCronJob={noopAsync}
        />
      </SidebarProvider>
    </MemoryRouter>
  )

  const { result } = renderHook(() => useSessionListActions({ profileScope: 'all' }))

  return async (response: SidebarSessionsResponse) => {
    listSidebarSessions.mockResolvedValueOnce(response)
    await act(async () => {
      await result.current.refreshSessions()
    })
  }
}

const groupIds = () =>
  [...window.document.querySelectorAll('[data-gateway-group]')].map(group => group.getAttribute('data-gateway-group'))

const group = (profile: string) =>
  window.document.querySelector(`[data-gateway-group='${JSON.stringify([null, profile])}']`) as HTMLElement | null

beforeEach(() => {
  listSidebarSessions.mockReset()
  setApiRequestConnection(null)
  $connectionsRegistry.set(null)
  $profiles.set([
    { name: 'default', is_default: true },
    { name: 'quiet', is_default: false },
    { name: 'work', is_default: false }
  ] as typeof $profiles.value)
  $showAllProfiles.set(true)
  setSidebarGrouping('profile')
})

afterEach(() => {
  cleanup()
  $showAllProfiles.set(false)
  $sessions.set([])
})

it('hides the empty section of a profile whose scan failed until a clean refresh reads it', async () => {
  const refresh = mountSidebar()

  await refresh(page([chat('one', 'default')], [{ error: 'database is locked', profile: 'work' }]))

  expect(groupIds().sort()).toEqual([JSON.stringify([null, 'default']), JSON.stringify([null, 'quiet'])])
  expect(within(group('quiet')!).getByText('No sessions yet')).toBeTruthy()
  expect(within(group('default')!).getByText('one title')).toBeTruthy()

  await refresh(page([chat('one', 'default')]))

  expect(groupIds().sort()).toEqual(
    [JSON.stringify([null, 'default']), JSON.stringify([null, 'quiet']), JSON.stringify([null, 'work'])].sort()
  )
  expect(within(group('work')!).getByText('No sessions yet')).toBeTruthy()
  expect(window.document.querySelectorAll('[data-gateway-group-empty]')).toHaveLength(2)
})

it('keeps the loaded rows of a profile whose next scan failed', async () => {
  const refresh = mountSidebar()

  await refresh(page([chat('one', 'default'), chat('kept', 'work')]))
  await refresh(page([chat('one', 'default')], [{ error: 'database is locked', profile: 'work' }]))

  expect(groupIds().sort()).toEqual(
    [JSON.stringify([null, 'default']), JSON.stringify([null, 'quiet']), JSON.stringify([null, 'work'])].sort()
  )
  expect(within(group('work')!).getByText('kept title')).toBeTruthy()
  expect(within(group('work')!).queryByText('No sessions yet')).toBeNull()
  expect(within(group('quiet')!).getByText('No sessions yet')).toBeTruthy()
})
