import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $apiRequestScope } from '@/api/client'
import * as apiClient from '@/api/client'
import { openSession } from '@/app/open-session'
import { group } from '@/components/pane-shell/tree/model'
import { $layoutTree, noteActiveTreeGroup } from '@/components/pane-shell/tree/store'
import { RootTooltipProvider } from '@/components/ui/tooltip'
import * as notifications from '@/store/notifications'
import { $activeGatewayProfile } from '@/store/profile'
import { $activeSessionId, $selectedStoredSessionId, $sessions, _resetSessionOwnerHintsForTests, mergeSessionPage } from '@/store/session'
import { $sessionTiles } from '@/store/session-states'
import { setPortalPreferred } from '@/store/shell-variant'
import { makeSessionInfo } from '@/test/session-info'

import { SidebarSessionRow } from './session-row'
import { SimplifiedSessionList } from './simplified-session-list'

const owner = { connectionId: 'remote-b', profile: 'research' }
const remote = makeSessionInfo({ id: 'search-only', connection_id: owner.connectionId, profile: owner.profile, title: 'Search-only remote' })

const noop = () => {}
const originalBridge = window.hermesDesktop

function mountRow(redesign: boolean) {
  setPortalPreferred(redesign)

  return render(
    <MemoryRouter><RootTooltipProvider>
      {redesign ? <SimplifiedSessionList
        activeSessionId={null} ambiguousIds={new Set()} onArchiveSession={noop} onBranchSession={noop}
        onDeleteSession={noop} onResumeSession={noop} onUpdate={noop} pinnedIds={new Set()} sessions={[remote]}
      /> : <SidebarSessionRow isPinned={false} isSelected={false} onArchive={noop} onDelete={noop}
        onPin={noop} onResume={noop} onToggleUnread={noop} session={remote} unread={false} />}
    </RootTooltipProvider></MemoryRouter>
  )
}

beforeEach(() => {
  $apiRequestScope.set({ connectionId: 'local', profile: 'default' })
  $activeGatewayProfile.set('default')
  $selectedStoredSessionId.set(null)
  $activeSessionId.set(null)
  $sessions.set([])
  $sessionTiles.set([])
  $layoutTree.set(null)
  noteActiveTreeGroup(null)
  _resetSessionOwnerHintsForTests()
})

afterEach(() => {
  cleanup()
  $sessionTiles.set([])
  $sessions.set([])
  $layoutTree.set(null)
  noteActiveTreeGroup(null)
  window.hermesDesktop = originalBridge
  $apiRequestScope.set({ connectionId: null, profile: null })
  setPortalPreferred(false)
  vi.restoreAllMocks()
})

describe.each([false, true])('session row navigation, redesign=%s', redesign => {
  it('carries an uncached remote search result into a new sessions tab', () => {
    mountRow(redesign)
    fireEvent.click(screen.getByText(remote.title!), { ctrlKey: true })
    expect($sessions.get()).toEqual([])
    expect($sessionTiles.get()).toMatchObject([{ storedSessionId: remote.id, workspaceMode: 'sessions', ownerRoute: owner }])
  })

  it('carries an uncached remote search result into the native window request', async () => {
    const open = vi.fn().mockResolvedValue({ ok: true })
    window.hermesDesktop = { ...originalBridge, openSessionWindow: open }
    mountRow(redesign)
    fireEvent.click(screen.getByText(remote.title!), { ctrlKey: true, shiftKey: true })
    await waitFor(() => expect(open).toHaveBeenCalledWith(remote.id, {
      connectionId: 'remote-b', profile: 'research', watch: undefined
    }))
    expect($sessionTiles.get()).toEqual([])
  })

  it('rejects a different-owner same-ID tab before touching its existing surface', () => {
    vi.spyOn(notifications, 'notifyError').mockReturnValue('owner-conflict')
    const request = vi.spyOn(apiClient, 'hermesApi')
    const otherOwner = { connectionId: 'local', profile: 'default' }
    const tile = { storedSessionId: remote.id, dir: 'center' as const, ownerRoute: otherOwner, runtimeId: 'local-runtime' }
    $sessionTiles.set([tile])
    const navigate = vi.fn()
    mountRow(redesign)
    fireEvent.click(screen.getByText(remote.title!), { ctrlKey: true })
    openSession(remote.id, navigate, 'in-place', { workspaceMode: 'sessions', ownerRoute: owner })
    expect($sessionTiles.get()).toEqual([tile])
    expect($sessionTiles.get()[0]).toBe(tile)
    expect(navigate).not.toHaveBeenCalled()
    expect(request).not.toHaveBeenCalled()
  })
})

it('highlights the focused background-owner tile, not its same-ID foreground twin', () => {
  const local = makeSessionInfo({ id: remote.id, connection_id: 'local', profile: 'default', title: 'Local twin' })
  $sessions.set([local, remote])
  $sessionTiles.set([{ storedSessionId: remote.id, dir: 'center', ownerRoute: owner }])
  $layoutTree.set(group([`session-tile:${remote.id}`], { id: 'focused', active: `session-tile:${remote.id}` }))
  noteActiveTreeGroup('focused')
  render(<MemoryRouter><RootTooltipProvider><SimplifiedSessionList
    activeSessionId={remote.id} ambiguousIds={new Set([remote.id])} onArchiveSession={noop} onBranchSession={noop}
    onDeleteSession={noop} onResumeSession={noop} onUpdate={noop} pinnedIds={new Set()} sessions={[local, remote]}
  /></RootTooltipProvider></MemoryRouter>)
  expect(screen.getByText(remote.title!).closest('button')?.getAttribute('aria-current')).toBe('page')
  expect(screen.getByText('Local twin').closest('button')?.hasAttribute('aria-current')).toBe(false)
  act(() => $apiRequestScope.set({ connectionId: 'another-gateway', profile: 'other' }))
  expect(screen.getByText(remote.title!).closest('button')?.getAttribute('aria-current')).toBe('page')
})

it('never merges a same-ID remote page into a retained local conversation', () => {
  const local = makeSessionInfo({ id: 'same', connection_id: 'local', profile: 'default', title: 'Local title', last_active: 100 })
  const other = makeSessionInfo({ id: 'same', connection_id: 'remote-b', profile: 'default', title: null, last_active: 1 })
  const merged = mergeSessionPage([local], [other], ['same'])
  expect(merged).toHaveLength(2)
  expect(merged.find(row => row.connection_id === 'remote-b')).toMatchObject({ title: null, last_active: 1 })
  expect(merged.find(row => row.connection_id === 'local')).toBe(local)
})
