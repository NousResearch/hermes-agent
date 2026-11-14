import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { $apiRequestScope } from '@/api/client'
import type { DesktopProfileRoute } from '@/global'
import { createProfile } from '@/hermes'
import type * as Hermes from '@/hermes'
import { en } from '@/i18n/en'
import { writeClipboardText } from '@/lib/clipboard'
import { $hiddenSidebarAgents, agentSidebarKey, setSidebarAgentHidden } from '@/store/agent-sidebar-preferences'
import { $defaultProfileRoute } from '@/store/default-profile'
import type * as FleetRoster from '@/store/fleet-roster'
import { $fleetRoster, refreshFleetRoster } from '@/store/fleet-roster'
import { notifyError } from '@/store/notifications'
import { $profiles, refreshProfiles } from '@/store/profile'
import type * as Profile from '@/store/profile'
import type { ProfileInfo } from '@/types/hermes'

import { AgentOverviewActions } from './overview-actions'

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<typeof Hermes>()),
  createProfile: vi.fn(async () => ({ soul_set: true }))
}))
vi.mock('@/store/profile', async importOriginal => ({
  ...(await importOriginal<typeof Profile>()),
  refreshProfiles: vi.fn(async () => undefined)
}))
vi.mock('@/store/fleet-roster', async importOriginal => ({
  ...(await importOriginal<typeof FleetRoster>()),
  refreshFleetRoster: vi.fn(async () => undefined)
}))
vi.mock('@/lib/clipboard', () => ({ writeClipboardText: vi.fn(async () => undefined) }))
vi.mock('@/store/notifications', () => ({ notify: vi.fn(), notifyError: vi.fn() }))

const owner = { connectionId: 'alpha', profile: 'research' }

const profile: ProfileInfo = {
  name: 'research',
  display_name: 'Research agent',
  path: '/srv/alpha/profiles/research',
  has_env: true,
  is_default: false,
  model: null,
  provider: null,
  skill_count: 0
}

const setDefault = vi.fn(async (route: DesktopProfileRoute) => route)

function openMenu() {
  const trigger = screen.getByRole('button', { name: 'Agent actions' })
  fireEvent.pointerDown(trigger, { button: 0, pointerType: 'mouse' })
  fireEvent.pointerUp(trigger, { button: 0, pointerType: 'mouse' })
  fireEvent.click(trigger)
}

beforeEach(() => {
  Object.defineProperty(window, 'hermesDesktop', {
    configurable: true,
    value: { profile: { setDefault } }
  })
  $apiRequestScope.set(owner)
  $profiles.set([profile])
  $hiddenSidebarAgents.set([])
  $defaultProfileRoute.set(null)
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
  $apiRequestScope.set({ profile: null, connectionId: null })
  $profiles.set([])
  $hiddenSidebarAgents.set([])
  $defaultProfileRoute.set(null)
  $fleetRoster.set(null)
  delete window.document.documentElement.dataset.hermesDesktopHost
})

it('reflects sidebar visibility live and changes only the actual profile on its own gateway', async () => {
  $fleetRoster.set({
    sources: [],
    agents: [
      {
        profile: 'research',
        targetProfile: 'default',
        connectionId: 'alpha',
        connectionKind: 'remote',
        connectionLabel: 'Alpha',
        handle: 'research-alpha'
      }
    ]
  })
  const target = { connectionId: 'alpha', profile: 'default' }
  const other = { connectionId: 'beta', profile: 'default' }
  setSidebarAgentHidden(other, true)
  render(<AgentOverviewActions owner={owner} profile={profile} />)
  openMenu()
  expect(screen.getByRole('menuitem', { name: 'Hide from chat list' })).toBeTruthy()
  act(() => setSidebarAgentHidden(target, true))
  await act(async () => fireEvent.click(screen.getByRole('menuitem', { name: 'Show in chat list' })))
  expect($hiddenSidebarAgents.get()).toEqual([agentSidebarKey(other)])
  openMenu()
  await act(async () => fireEvent.click(screen.getByRole('menuitem', { name: 'Hide from chat list' })))
  expect($hiddenSidebarAgents.get()).toContain(agentSidebarKey(target))
  expect($hiddenSidebarAgents.get()).not.toContain(agentSidebarKey(owner))
  expect(createProfile).not.toHaveBeenCalled()
  expect(setDefault).not.toHaveBeenCalled()
})

it('shows and copies the real profile path, and starts deleting from the menu itself', async () => {
  render(<AgentOverviewActions owner={owner} profile={profile} />)
  openMenu()
  const copy = screen.getByRole('menuitem', { name: /Copy agent folder path/ })
  // One line: the label, then the path as a trailing hint.
  expect(copy.textContent).toBe(`Copy agent folder path${profile.path}`)
  await act(async () => fireEvent.click(screen.getByRole('menuitem', { name: /^Delete agent/ })))
  expect(screen.getByRole('dialog', { name: en.profiles.deleteTitle })).toBeTruthy()
  fireEvent.click(screen.getByRole('button', { name: en.common.cancel }))
  openMenu()
  await act(async () => fireEvent.click(screen.getByRole('menuitem', { name: /Copy agent folder path/ })))
  expect(writeClipboardText).toHaveBeenCalledWith(profile.path)
  vi.mocked(writeClipboardText).mockRejectedValueOnce(new Error('Clipboard unavailable'))
  openMenu()
  await act(async () => fireEvent.click(screen.getByRole('menuitem', { name: /Copy agent folder path/ })))
  expect(notifyError).toHaveBeenCalledWith(expect.any(Error), en.common.copyFailed)
})

it('duplicates the captured profile on the captured gateway even after the active gateway changes', async () => {
  render(<AgentOverviewActions owner={owner} profile={profile} />)
  openMenu()
  await act(async () => fireEvent.click(screen.getByRole('menuitem', { name: 'Duplicate agent' })))
  act(() => $apiRequestScope.set({ connectionId: 'beta', profile: 'other' }))
  fireEvent.change(screen.getByRole('textbox', { name: en.profiles.nameLabel }), {
    target: { value: 'research-copy' }
  })
  await act(async () => fireEvent.click(screen.getByRole('button', { name: en.profiles.createAction })))
  expect(createProfile).toHaveBeenCalledWith({ name: 'research-copy', clone_from: 'research' }, owner)
  expect(refreshProfiles).not.toHaveBeenCalled()
  expect(refreshFleetRoster).toHaveBeenCalledWith({ force: true })
})

it('titles the duplicate dialog by the action that opened it, not "New agent"', async () => {
  render(<AgentOverviewActions owner={owner} profile={profile} />)
  openMenu()
  await act(async () => fireEvent.click(screen.getByRole('menuitem', { name: 'Duplicate agent' })))

  expect(screen.getByRole('dialog', { name: 'Duplicate agent' })).toBeTruthy()
  expect(screen.queryByRole('heading', { name: en.profiles.newProfile })).toBeNull()
})

it('sets the owner route as default, not a same-named agent on the active gateway', async () => {
  $defaultProfileRoute.set({ connectionId: 'beta', profile: 'research' })
  $apiRequestScope.set({ connectionId: 'beta', profile: 'research' })
  render(<AgentOverviewActions owner={owner} profile={profile} />)
  openMenu()
  await act(async () => fireEvent.click(screen.getByRole('menuitem', { name: 'Set as default agent' })))
  expect(setDefault).toHaveBeenCalledWith(owner)
  expect($defaultProfileRoute.get()).toEqual(owner)
  openMenu()
  expect(screen.getByRole('menuitem', { name: en.profiles.defaultProfile }).getAttribute('aria-disabled')).toBe('true')
})

it('explicitly disables unsupported default and missing profile capabilities', () => {
  window.document.documentElement.dataset.hermesDesktopHost = 'browser'
  render(<AgentOverviewActions owner={owner} />)
  openMenu()
  expect(screen.getByRole('menuitem', { name: 'Set as default agent' }).getAttribute('aria-disabled')).toBe('true')
  expect(screen.getByText('Setting a default agent is unavailable for this machine in this app.')).toBeTruthy()
  expect(screen.getByRole('menuitem', { name: 'Duplicate agent' }).getAttribute('aria-disabled')).toBe('true')
  expect(screen.getByRole('menuitem', { name: /Copy agent folder path/ }).getAttribute('aria-disabled')).toBe('true')
  expect(setDefault).not.toHaveBeenCalled()
})
