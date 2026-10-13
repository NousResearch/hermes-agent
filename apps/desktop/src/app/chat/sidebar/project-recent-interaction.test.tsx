import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, it } from 'vitest'

import { I18nProvider, TRANSLATIONS } from '@/i18n'
import {
  $sidebarProjectOrderIds,
  $sidebarProjectRecentScopes,
  resetSidebarView,
  setSidebarGrouping
} from '@/store/layout'
import { $activeGatewayProfile, $showAllProfiles } from '@/store/profile'
import { $connection } from '@/store/session'

import { SidebarFilterMenu } from './filter-menu'
import { projectSortModeForScope, projectSortScopeKey, sortProjectsByRecentActivity } from './projects/activity-sort'

const f = TRANSLATIONS.en.sidebar.filter
const recent = TRANSLATIONS.en.shell.gatewayMenu.recentActivity
const scope = (connection: string, profile: string) => projectSortScopeKey(connection, profile)

const mode = (connection: string | null, profile: string) =>
  projectSortModeForScope($sidebarProjectRecentScopes.get(), connection, profile)

function openProjectOrdering() {
  fireEvent.keyDown(screen.getByRole('button', { name: f.filters }), { key: 'Enter' })
  fireEvent.keyDown(screen.getByRole('menuitem', { name: `${f.project} ${f.ordering}` }), { key: 'ArrowRight' })
}

afterEach(() => {
  cleanup()
  $sidebarProjectRecentScopes.set({})
  $sidebarProjectOrderIds.set([])
  $showAllProfiles.set(false)
  $activeGatewayProfile.set('default')
  $connection.set(null)
  resetSidebarView()
})

it('toggles opt-in recent and manual through the rendered menu, preserving saved drag order and other scopes', () => {
  $connection.set({ connectionId: 'A', baseUrl: 'http://fixture.invalid', mode: 'remote' } as never)
  $sidebarProjectOrderIds.set(['older', 'newer'])
  setSidebarGrouping('project')
  render(
    <I18nProvider configClient={null} initialLocale="en">
      <SidebarFilterMenu />
    </I18nProvider>
  )
  openProjectOrdering()
  expect(screen.getByRole('menuitemradio', { name: f.manual }).getAttribute('aria-checked')).toBe('true')
  fireEvent.click(screen.getByRole('menuitemradio', { name: recent }))
  expect(mode('A', 'default')).toBe('recent')
  expect(screen.getByRole('menuitemradio', { name: recent }).getAttribute('aria-checked')).toBe('true')
  expect($sidebarProjectOrderIds.get()).toEqual(['older', 'newer'])

  act(() => $activeGatewayProfile.set('other'))
  expect(mode('A', 'other')).toBe('manual')
  expect(screen.getByRole('menuitemradio', { name: f.manual }).getAttribute('aria-checked')).toBe('true')
  act(() => $activeGatewayProfile.set('default'))
  expect(mode('A', 'default')).toBe('recent')
  act(() => $connection.set({ connectionId: 'B', baseUrl: 'http://fixture.invalid', mode: 'remote' } as never))
  expect(mode('B', 'default')).toBe('manual')
  act(() => $connection.set({ connectionId: 'A', baseUrl: 'http://fixture.invalid', mode: 'remote' } as never))
  expect(mode('A', 'default')).toBe('recent')
  fireEvent.click(screen.getByRole('menuitemradio', { name: f.manual }))
  expect(mode('A', 'default')).toBe('manual')
  expect($sidebarProjectOrderIds.get()).toEqual(['older', 'newer'])
})

it('only exposes project ordering in connected project grouping; sorts qualified messages, never heartbeat or tool activity', () => {
  setSidebarGrouping('date')
  render(
    <I18nProvider configClient={null} initialLocale="en">
      <SidebarFilterMenu />
    </I18nProvider>
  )
  fireEvent.keyDown(screen.getByRole('button', { name: f.filters }), { key: 'Enter' })
  expect(screen.queryByRole('menuitem', { name: `${f.project} ${f.ordering}` })).toBeNull()

  const rows = [
    { id: 'heartbeat-tool', label: 'A', sessionCount: 1, lastActive: 900, lastMessageAt: 100 },
    { id: 'missing', label: 'C', sessionCount: 1, lastActive: 990 },
    { id: 'user-assistant', label: 'B', sessionCount: 1, lastActive: 800, lastMessageAt: 800 }
  ] as never

  expect(sortProjectsByRecentActivity(rows).map((row: { id: string }) => row.id)).toEqual([
    'user-assistant',
    'heartbeat-tool',
    'missing'
  ])
  expect(mode(null, 'default')).toBe('manual')
})

it('orders a historical project with zero selected rows by its uncapped message clock', () => {
  const rows = [
    { id: 'busy', label: 'Busy', sessionCount: 2000, lastMessageAt: 0, lastActive: 900 },
    { id: 'old', label: 'Old', sessionCount: 0, lastMessageAt: 710, lastActive: 0 },
    { id: 'empty', label: 'Empty', sessionCount: 0, lastMessageAt: 0, lastActive: 999 },
    { id: 'home', label: 'Home', isNoProject: true, sessionCount: 0, lastMessageAt: 0 }
  ] as never

  expect(sortProjectsByRecentActivity(rows).map((row: { id: string }) => row.id)).toEqual([
    'home',
    'old',
    'busy',
    'empty'
  ])
})
