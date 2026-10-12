import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, expect, it } from 'vitest'

import { defaultNewSessionTarget } from '@/app/session/new-session-route'
import { I18nProvider, TRANSLATIONS, useI18n } from '@/i18n'
import type { I18nContextValue } from '@/i18n'
import { $interfaceMode } from '@/store/interface-mode'
import {
  $sidebarListGroupIds,
  $sidebarOrdering,
  $sidebarProfileFilter,
  $sidebarStatusFilter,
  resetSidebarView,
  setSidebarGrouping,
  setSidebarOrdering,
  toggleSidebarProfileFilter
} from '@/store/layout'
import { $activeGatewayProfile, $freshSessionRequest, $profiles } from '@/store/profile'
import { $selectedStoredSessionId } from '@/store/session'

import { SidebarFilterMenu } from './filter-menu'

let i18n: I18nContextValue

function Menu() {
  i18n = useI18n()

  return <SidebarFilterMenu />
}

afterEach(() => {
  cleanup()
  $interfaceMode.set('advanced')
  resetSidebarView()
})

it('translates the live filter menu while preserving selected values and the current grouping order', async () => {
  $interfaceMode.set('advanced')
  setSidebarGrouping('date')
  setSidebarOrdering('manual')
  $sidebarListGroupIds.set(['today'])
  render(
    <I18nProvider configClient={null} initialLocale="zh">
      <MemoryRouter>
        <Menu />
      </MemoryRouter>
    </I18nProvider>
  )
  const zh = TRANSLATIONS.zh.sidebar.filter
  fireEvent.keyDown(screen.getByRole('button', { name: zh.filters }), { key: 'Enter' })
  const group = screen.getByRole('menuitem', { name: new RegExp(zh.grouping) })
  fireEvent.keyDown(group, { key: 'ArrowRight' })
  expect(screen.getByRole('menuitemradio', { name: zh.updated }).getAttribute('aria-checked')).toBe('true')
  fireEvent.keyDown(screen.getByRole('menuitemradio', { name: zh.updated }), { key: 'Escape' })
  fireEvent.keyDown(screen.getByRole('button', { name: zh.filters }), { key: 'Enter' })
  fireEvent.keyDown(screen.getByRole('menuitem', { name: zh.status }), { key: 'ArrowRight' })
  fireEvent.click(screen.getByRole('menuitemcheckbox', { name: zh.needsInput }))
  expect($sidebarStatusFilter.get()).toContain('needs-input')
  await act(() => i18n.setLocale('ja'))
  const ja = TRANSLATIONS.ja.sidebar.filter
  expect(screen.getByText(TRANSLATIONS.ja.sidebar.profileRail)).toBeTruthy()
  expect(screen.getByText(TRANSLATIONS.ja.sidebar.markAllRead)).toBeTruthy()
  expect(screen.queryByText(TRANSLATIONS.en.sidebar.profileRail)).toBeNull()
  expect(screen.queryByText(TRANSLATIONS.en.sidebar.markAllRead)).toBeNull()
  fireEvent.keyDown(screen.getByRole('menuitem', { name: ja.status }), { key: 'ArrowRight' })
  expect(screen.getByRole('menuitemcheckbox', { name: ja.needsInput }).getAttribute('aria-checked')).toBe('true')
  expect(screen.queryByRole('menuitemcheckbox', { name: zh.needsInput })).toBeNull()
  expect($sidebarOrdering.get()).toBe('manual')
})

it('uses one profile control in Simple and preserves the Advanced profile subset', () => {
  $interfaceMode.set('advanced')
  toggleSidebarProfileFilter('research')
  expect($sidebarProfileFilter.get()).toEqual(['research'])
  $interfaceMode.set('simple')
  expect($sidebarProfileFilter.get()).toEqual([])
  render(
    <MemoryRouter>
      <SidebarFilterMenu label="Organize chats" side="top" />
    </MemoryRouter>
  )
  fireEvent.keyDown(screen.getByRole('button', { name: 'Organize chats' }), { key: 'Enter' })
  expect(screen.queryByRole('menuitem', { name: 'Profile' })).toBeNull()
  expect(screen.queryByRole('menuitemcheckbox', { name: 'All profiles' })).toBeNull()
  expect(screen.getByRole('menuitem', { name: /^Grouping/ })).toBeTruthy()
  expect(screen.getByRole('menuitem', { name: 'Status' })).toBeTruthy()
  act(() => $interfaceMode.set('advanced'))
  expect($sidebarProfileFilter.get()).toEqual(['research'])
  expect(screen.getByRole('menuitem', { name: 'Profile' })).toBeTruthy()
})

it('filters history without leaving the chat and uses the single filtered profile for a new chat', () => {
  $interfaceMode.set('simple')
  $activeGatewayProfile.set('default')
  $profiles.set([
    { name: 'default', is_default: true },
    { name: 'research', is_default: false }
  ] as typeof $profiles.value)
  $selectedStoredSessionId.set('existing-chat')
  const freshBefore = $freshSessionRequest.get()
  render(
    <MemoryRouter>
      <SidebarFilterMenu label="Organize chats" />
    </MemoryRouter>
  )
  fireEvent.keyDown(screen.getByRole('button', { name: 'Organize chats' }), { key: 'Enter' })
  fireEvent.keyDown(screen.getByRole('menuitem', { name: 'Profile' }), { key: 'ArrowRight' })
  fireEvent.click(screen.getByRole('menuitemcheckbox', { name: 'research' }))
  expect($sidebarProfileFilter.get()).toEqual(['research'])
  expect($selectedStoredSessionId.get()).toBe('existing-chat')
  expect($freshSessionRequest.get()).toBe(freshBefore)
  expect(defaultNewSessionTarget()?.profile).toBe('research')
  fireEvent.click(screen.getByRole('menuitemcheckbox', { name: 'All profiles' }))
  expect(defaultNewSessionTarget()?.profile).toBe('default')
  $selectedStoredSessionId.set(null)
  $profiles.set([])
})
