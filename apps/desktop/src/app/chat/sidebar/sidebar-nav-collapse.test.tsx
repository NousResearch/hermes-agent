// @vitest-environment jsdom
//
// The nav block (New session → Themes: built-ins plus every plugin-contributed
// row) collapses as ONE unit — that is the contract, so a contributed row can
// never stay behind and strand the block half-hidden. The toggle itself must
// survive the collapse in both directions: it is the only way back.
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { group, split } from '@/components/pane-shell/tree/model'
import { $layoutTree, noteActiveTreeGroup } from '@/components/pane-shell/tree/store'
import { SidebarProvider } from '@/components/ui/sidebar'
import { registry } from '@/contrib/registry'
import { $sidebarNavCollapsed } from '@/store/layout'
import { $selectedStoredSessionId, $sessions } from '@/store/session'
import { $removedSessionIds } from '@/store/session-removal'
import { makeSessionInfo } from '@/test/session-info'

import { ROUTES_AREA, SIDEBAR_NAV_AREA } from '../../routes'

import { ChatSidebar } from './index'

const NAV_COLLAPSED_KEY = 'hermes.desktop.sidebarNavCollapsed'

const noop = () => {}

const noopAsync = async () => {}

// Rows are `sidebar-nav-<id>` — the durable handle the tip catalog points at,
// and a namespace nothing else in the sidebar wears.
const navRowHandles = (container: HTMLElement) =>
  Array.from(container.querySelectorAll('[data-tour^="sidebar-nav-"]')).map(el => el.getAttribute('data-tour'))

const navToggle = () => screen.getByRole('button', { name: 'Menu' })

const renderSidebar = () =>
  render(
    <MemoryRouter initialEntries={['/']}>
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
          onTriggerCronJob={noopAsync}
        />
      </SidebarProvider>
    </MemoryRouter>
  )

describe('sidebar nav collapse', () => {
  let disposeContributions: () => void

  beforeEach(() => {
    window.localStorage.clear()
    $sidebarNavCollapsed.set(false)
    // One contributed row (the shape the kanban plugin registers) so the test
    // covers "built-ins AND plugin rows" rather than the built-ins alone.
    disposeContributions = registry.registerMany([
      { area: ROUTES_AREA, id: 'kanban-page', data: { path: '/kanban' }, render: () => null },
      { area: SIDEBAR_NAV_AREA, id: 'kanban-nav', data: { codicon: 'project', label: 'Kanban', path: '/kanban' } }
    ])
    $selectedStoredSessionId.set('tile-one')
    $sessions.set([
      makeSessionInfo({ id: 'tile-one', last_active: 2, profile: 'default', started_at: 1, title: 'Tile one' })
    ])
    $removedSessionIds.set(new Set())
    $layoutTree.set(split('row', [group(['workspace'], { active: 'workspace', id: 'workspace-group' })]))
    noteActiveTreeGroup('workspace-group')
  })

  afterEach(() => {
    cleanup()
    disposeContributions()
    $sidebarNavCollapsed.set(false)
    window.localStorage.clear()
    $selectedStoredSessionId.set(null)
    $sessions.set([])
    $removedSessionIds.set(new Set())
    $layoutTree.set(null)
    noteActiveTreeGroup(null)
  })

  it('hides every nav row — built-in and contributed — leaving the toggle and the sessions list', () => {
    const { container } = renderSidebar()

    // Expanded: the built-ins and the contributed row are all on screen.
    for (const handle of [
      'sidebar-nav-new-session',
      'sidebar-nav-skills',
      'sidebar-nav-messaging',
      'sidebar-nav-artifacts',
      'sidebar-nav-cron'
    ]) {
      expect(navRowHandles(container)).toContain(handle)
    }

    expect(screen.getByRole('button', { name: /Kanban/ })).toBeTruthy()
    expect(screen.getByText('Tile one')).toBeTruthy()

    fireEvent.click(navToggle())

    // Collapsed: not one row left, from either source.
    expect(navRowHandles(container)).toEqual([])
    // The way back out stays on screen, and says which way it goes.
    expect(navToggle().getAttribute('aria-expanded')).toBe('false')
    // The list the space was given to is untouched by the collapse.
    expect(screen.getByText('Tile one')).toBeTruthy()
    expect(window.localStorage.getItem(NAV_COLLAPSED_KEY)).toBe('true')

    fireEvent.click(navToggle())

    expect(navRowHandles(container)).toContain('sidebar-nav-new-session')
    expect(navToggle().getAttribute('aria-expanded')).toBe('true')
    expect(window.localStorage.getItem(NAV_COLLAPSED_KEY)).toBe('false')
  })

  it('parks focus on the toggle instead of letting a row it unmounts drop it to the body', () => {
    renderSidebar()
    const row = screen.getByRole('button', { name: /Capabilities/ })

    row.focus()
    expect(globalThis.document.activeElement).toBe(row)

    fireEvent.click(navToggle())

    expect(globalThis.document.activeElement).toBe(navToggle())
  })
})
