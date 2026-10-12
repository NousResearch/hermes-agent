// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, expect, it, vi } from 'vitest'

import { SidebarProvider } from '@/components/ui/sidebar'
import { setInterfaceMode, setModeContext } from '@/store/interface-mode'
import { $activeGatewayProfile, $profiles } from '@/store/profile'
import { $profileRailVisible } from '@/store/profile-rail-prefs'
import {
  $remoteOverrideDialogProfile,
  closeRemoteOverrideDialog,
  openRemoteOverrideDialog
} from '@/store/profile-remote-override'
import { $sessions, $sessionsLoading } from '@/store/session'
import { $unreadFinishedMarkers } from '@/store/session-unread'

import { ProfileRemoteOverrideHost } from './profile-remote-override-dialog'

import { ChatSidebar } from './index'

const noop = () => {}

const renderSidebar = () =>
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
          onRetrySessions={async () => {}}
          onTriggerCronJob={async () => {}}
        />
      </SidebarProvider>
    </MemoryRouter>
  )

afterEach(() => {
  cleanup()
  closeRemoteOverrideDialog()
  $unreadFinishedMarkers.set({})
  setInterfaceMode('advanced')
  vi.restoreAllMocks()
})

it('Simple preserves inactive-profile attention state in its closed organizer', () => {
  $activeGatewayProfile.set('default')
  $profiles.set(
    ['default', 'writer'].map(name => ({
      name,
      is_default: name === 'default',
      path: '',
      has_env: false,
      model: '',
      provider: '',
      skill_count: 0
    }))
  )
  $sessions.set([])
  $sessionsLoading.set(false)
  setModeContext({ profileCount: 2, connectionCount: 1 })
  setInterfaceMode('advanced')
  $profileRailVisible.set(true)
  $unreadFinishedMarkers.set({ writer: ['background-finished'] })
  const view = renderSidebar()
  expect(screen.getByRole('button', { name: /writer,.*1/ })).toBeTruthy()
  expect(view.container.querySelector('[data-slot="profile-status-dot"]')).not.toBeNull()
  act(() => setInterfaceMode('simple'))
  expect(view.container.querySelector('[data-slot="profile-status-dot"]')).not.toBeNull()
  const dropdown = screen.getByRole('button', { name: /Organize chats.*1 unread/ })
  fireEvent.pointerDown(dropdown, { button: 0, pointerType: 'mouse', ctrlKey: false })
  fireEvent.keyDown(screen.getByRole('menuitem', { name: 'Profile' }), { key: 'ArrowRight' })
  expect(screen.getByRole('menuitemcheckbox', { name: /writer.*1 unread/ })).toBeTruthy()
})

it('Simple honors an effective rail reveal', () => {
  setInterfaceMode('simple')
  setModeContext({ profileCount: 1, connectionCount: 1 })
  const view = renderSidebar()
  act(() => $profileRailVisible.set(true))
  expect($profileRailVisible.get()).toBe(true)
  expect(view.container.querySelector('[data-slot="profile-rail"]')).not.toBeNull()
  act(() => setInterfaceMode('advanced'))
  act(() => $profileRailVisible.set(true))
  expect(view.container.querySelector('[data-slot="profile-rail"]')).not.toBeNull()
})

it('remote-profile token repair remains available in Simple without a sidebar', () => {
  setInterfaceMode('simple')
  const host = render(<ProfileRemoteOverrideHost />)
  act(() => openRemoteOverrideDialog('writer'))
  expect($remoteOverrideDialogProfile.get()).toBe('writer')
  expect(screen.getByRole('dialog')).toBeTruthy()
  act(() => setInterfaceMode('advanced'))
  expect(screen.getAllByRole('dialog')).toHaveLength(1)
  host.unmount()
})
