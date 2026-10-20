import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, it } from 'vitest'

import { I18nProvider, TRANSLATIONS, useI18n } from '@/i18n'
import type { I18nContextValue } from '@/i18n'
import { $interfaceMode } from '@/store/interface-mode'
import {
  $sidebarListGroupIds,
  $sidebarOrdering,
  $sidebarRowMeta,
  $sidebarStatusFilter,
  resetSidebarView,
  setSidebarGrouping,
  setSidebarOrdering
} from '@/store/layout'

import { SidebarFilterMenu } from './filter-menu'

let i18n: I18nContextValue

function Menu() {
  i18n = useI18n()

  return <SidebarFilterMenu />
}

afterEach(() => {
  cleanup()
  resetSidebarView()
})

it('translates the live filter menu while preserving selected values and the current grouping order', async () => {
  $interfaceMode.set('advanced')
  setSidebarGrouping('date')
  setSidebarOrdering('manual')
  $sidebarListGroupIds.set(['today'])
  render(
    <I18nProvider configClient={null} initialLocale="zh">
      <Menu />
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

it('offers no checkbox for the always-on profile chip, and still decodes a persisted one', async () => {
  $interfaceMode.set('advanced')
  // A value written by an older build, before the chip became unconditional.
  $sidebarRowMeta.set(['updated', 'preview', 'profile'])
  render(
    <I18nProvider configClient={null} initialLocale="en">
      <Menu />
    </I18nProvider>
  )
  const f = TRANSLATIONS.en.sidebar.filter
  fireEvent.keyDown(screen.getByRole('button', { name: f.filters }), { key: 'Enter' })

  // The row-details submenu is gated on showsAdvancedChrome only; its options
  // are the ROW_META entries. 'profile' must not be among them, since the chip
  // renders unconditionally and a switch here would change nothing on screen.
  fireEvent.keyDown(screen.getByRole('menuitem', { name: f.show }), { key: 'ArrowRight' })
  expect(screen.queryByRole('menuitemcheckbox', { name: f.profile })).toBeNull()
  // Its siblings are still present, so the assertion above is about 'profile'
  // specifically and not an empty submenu.
  expect(screen.getByRole('menuitemcheckbox', { name: f.updated })).toBeTruthy()

  // The stored value survives the codec round-trip rather than being stripped:
  // the union member is still legal input, just unread.
  await act(async () => {
    $sidebarRowMeta.set($sidebarRowMeta.get())
  })
  expect($sidebarRowMeta.get()).toContain('profile')
})
