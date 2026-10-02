import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, it } from 'vitest'

import { I18nProvider, TRANSLATIONS, useI18n } from '@/i18n'
import type { I18nContextValue } from '@/i18n'
import { $interfaceMode } from '@/store/interface-mode'
import {
  $sidebarListGroupIds,
  $sidebarOrdering,
  $sidebarStatusFilter,
  resetSidebarView,
  setSidebarGrouping,
  setSidebarOrdering
} from '@/store/layout'
import { $sessions, $unreadFinishedSessionIds, setSessions } from '@/store/session'
import { $sessionSeenCounts, $unreadFinishedMarkers } from '@/store/session-unread'
import { makeSessionInfo } from '@/test/session-info'

import { SidebarFilterMenu } from './filter-menu'

let i18n: I18nContextValue

function Menu() {
  i18n = useI18n()

  return <SidebarFilterMenu />
}

afterEach(() => {
  cleanup()
  resetSidebarView()
  // Lists first: their listeners recompute the unread atom, so wiping the
  // atom afterwards leaves a genuinely clean slate.
  $sessions.set([])
  $sessionSeenCounts.set({})
  $unreadFinishedMarkers.set({})
  $unreadFinishedSessionIds.set([])
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

it('mark-all-as-read acks the persisted layer so a later refresh does not repaint', () => {
  // A row past its seen-watermark is unread: the paint atom is rebuilt from
  // this persisted gap on every list refresh (this is the Cmd-R / restart path).
  $sessionSeenCounts.set({ default: { s1: 3 } })
  setSessions([makeSessionInfo({ id: 's1', message_count: 5 })])
  expect($unreadFinishedSessionIds.get()).toContain('s1')

  render(
    <I18nProvider configClient={null}>
      <Menu />
    </I18nProvider>
  )
  const en = TRANSLATIONS.en.sidebar
  fireEvent.keyDown(screen.getByRole('button', { name: en.filter.filters }), { key: 'Enter' })
  fireEvent.click(screen.getByRole('menuitem', { name: en.markAllRead }))

  // The dots are gone AND the persisted watermark caught up to the live count
  // — otherwise the very next refresh rebuilds every dot just dismissed.
  expect($unreadFinishedSessionIds.get()).toEqual([])
  expect($sessionSeenCounts.get()).toEqual({ default: { s1: 5 } })

  // "Refresh": the same rows arriving in a new list must stay read.
  setSessions([makeSessionInfo({ id: 's1', message_count: 5 })])
  expect($unreadFinishedSessionIds.get()).toEqual([])
})
