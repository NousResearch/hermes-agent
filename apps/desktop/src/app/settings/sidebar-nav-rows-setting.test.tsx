// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen, within } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { SIDEBAR_NAV_AREA } from '@/app/routes'
import { registry } from '@/contrib/registry'
import { en } from '@/i18n/en'
import { $sidebarNavHidden, setSidebarNavHidden, SIDEBAR_NAV_IDS } from '@/store/sidebar-nav'

import { SETTINGS_MANIFEST } from './settings-manifest'
import { SidebarNavRowsSetting } from './sidebar-nav-rows-setting'

const a = en.settings.appearance
const NAV_LABELS: Record<string, string> = en.sidebar.nav
// The component's built-in list IS the store's canonical list — sourced from
// it, not re-literalled, so these assertions track the binding, not a copy.
const BUILT_INS: readonly string[] = SIDEBAR_NAV_IDS

const disposers: Array<() => void> = []

afterEach(() => {
  cleanup()
  disposers.splice(0).forEach(dispose => dispose())
})

describe('SidebarNavRowsSetting', () => {
  beforeEach(() => {
    window.localStorage.clear()
    $sidebarNavHidden.set([])
  })

  it('lists all five built-in rows, checked when nothing is hidden', () => {
    render(<SidebarNavRowsSetting />)

    // ALL five, even the tier:'advanced' rows a Simple-mode user never sees:
    // the toggle persists for when the mode shows them again.
    const checkboxes = screen.getAllByRole('checkbox')
    expect(checkboxes).toHaveLength(BUILT_INS.length)

    for (const id of BUILT_INS) {
      const box = screen.getByRole('checkbox', { name: NAV_LABELS[id] })
      expect(box.getAttribute('aria-checked')).toBe('true')
    }
  })

  it('renders hidden rows unchecked and visible rows checked', () => {
    setSidebarNavHidden(['messaging', 'cron'])
    render(<SidebarNavRowsSetting />)

    for (const id of BUILT_INS) {
      const box = screen.getByRole('checkbox', { name: NAV_LABELS[id] })
      expect(box.getAttribute('aria-checked')).toBe(id === 'messaging' || id === 'cron' ? 'false' : 'true')
    }
  })

  it('toggling a checkbox flips the persisted hidden set in both directions', () => {
    render(<SidebarNavRowsSetting />)

    fireEvent.click(screen.getByRole('checkbox', { name: NAV_LABELS.artifacts }))
    expect($sidebarNavHidden.get()).toEqual(['artifacts'])

    fireEvent.click(screen.getByRole('checkbox', { name: NAV_LABELS.artifacts }))
    expect($sidebarNavHidden.get()).toEqual([])
  })

  it('re-renders from the store when a peer hides a row', () => {
    render(<SidebarNavRowsSetting />)
    const messaging = screen.getByRole('checkbox', { name: NAV_LABELS.messaging })

    act(() => setSidebarNavHidden(['messaging']))

    expect(messaging.getAttribute('aria-checked')).toBe('false')
    // The click still un-hides — the checkbox mirrors the store, not a local echo.
    fireEvent.click(messaging)
    expect($sidebarNavHidden.get()).toEqual([])
  })

  // Contributed rows come from the live registry (same area the sidebar reads),
  // keyed by the namespaced contribution id the store already tolerates.
  it('appends contributed nav rows with their payload label and namespaced id', () => {
    act(() => {
      disposers.push(
        registry.register({
          area: SIDEBAR_NAV_AREA,
          data: { codicon: 'project', label: 'Kanban', path: '/kanban' },
          id: 'kanban:nav',
          source: 'kanban'
        })
      )
    })

    render(<SidebarNavRowsSetting />)

    const kanban = screen.getByRole('checkbox', { name: 'Kanban' })
    expect(kanban.getAttribute('aria-checked')).toBe('true')
    expect(within(kanban.closest('label')!).getByText('Kanban')).toBeTruthy()

    fireEvent.click(kanban)
    expect($sidebarNavHidden.get()).toEqual(['kanban:nav'])
  })

  // A contribution the sidebar itself would not render (no route, no label)
  // must not offer a toggle for a row that can never exist.
  it('ignores contributed entries the sidebar would not render', () => {
    act(() => {
      disposers.push(
        registry.register({
          area: SIDEBAR_NAV_AREA,
          data: { codicon: 'plug', label: 'Pathless', path: 'not-a-route' },
          id: 'broken:pathless',
          source: 'broken'
        })
      )
    })

    render(<SidebarNavRowsSetting />)

    expect(screen.queryByRole('checkbox', { name: 'Pathless' })).toBeNull()
    expect(screen.getAllByRole('checkbox')).toHaveLength(BUILT_INS.length)
  })
})

// Contract, not snapshot: the row must exist, live beside Interface mode, and
// be searchable by the words a user reaches it with.
describe('settings manifest: appearance.sidebarNav', () => {
  const entry = SETTINGS_MANIFEST.appearance.sidebarNav

  it('lives on the window-layout subpage with row-hiding keywords', () => {
    expect(entry.subpage).toBe('window-layout')
    expect(entry.keywords).toEqual(
      expect.arrayContaining(['sidebar', 'navigation', 'hide', 'show', 'scheduled jobs', 'plugin'])
    )
  })

  it('resolves its label/description from the appearance catalog', () => {
    expect(entry.copy(en as never)).toEqual({
      description: a.sidebarNavDesc,
      label: a.sidebarNavTitle
    })
  })
})
