import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter, useNavigate } from 'react-router'
import { afterEach, beforeEach, expect, it } from 'vitest'

import { group, split } from '@/components/pane-shell/tree/model'
import {
  $hiddenStripTabs,
  $hiddenTreePanes,
  $layoutTree,
  noteActiveTreeGroup,
  setTreePaneHidden
} from '@/components/pane-shell/tree/store'
import { $routeCollapsedPanes } from '@/components/pane-shell/tree/store-route-collapse'
import { RootTooltipProvider } from '@/components/ui/tooltip'
import { $sidebarOpen, setSidebarOpen } from '@/store/layout'
import { $shellChoice } from '@/store/shell-variant'

import { SimplifiedNavigationSurface } from './simplified-navigation-surface'

function Surface() {
  const navigate = useNavigate()

  return (
    <SimplifiedNavigationSurface
      onNavigate={item => {
        if (item.route) {
          navigate(item.route)
        }
      }}
    />
  )
}

function mount(route = '/') {
  return render(
    <MemoryRouter initialEntries={[route]}>
      <RootTooltipProvider>
        <Surface />
      </RootTooltipProvider>
    </MemoryRouter>
  )
}

beforeEach(() => {
  $shellChoice.set(true)
  $hiddenStripTabs.set(new Set())
  $hiddenTreePanes.set(new Set())
  $routeCollapsedPanes.set(new Set())
  $layoutTree.set(null)
  noteActiveTreeGroup(null)
  setSidebarOpen(true)
})

afterEach(() => {
  cleanup()
  $shellChoice.set(false)
  $hiddenStripTabs.set(new Set())
  $hiddenTreePanes.set(new Set())
  $routeCollapsedPanes.set(new Set())
  $layoutTree.set(null)
  noteActiveTreeGroup(null)
  setSidebarOpen(true)
})

it('keeps global navigation while the sidebar is closed and preserves that choice across pages', () => {
  mount()
  act(() => setSidebarOpen(false))
  expect(screen.getByRole('navigation', { name: 'Hermes' })).toBeTruthy()
  fireEvent.click(screen.getByRole('button', { name: 'Capabilities' }))
  expect($routeCollapsedPanes.get().has('sessions')).toBe(true)
  expect($sidebarOpen.get()).toBe(false)
  fireEvent.click(screen.getByRole('button', { name: 'Chats' }))
  expect($routeCollapsedPanes.get().has('sessions')).toBe(false)
  expect($sidebarOpen.get()).toBe(false)
})

it('collapses sessions on global pages without hiding them, so an attempted reveal cannot show them', () => {
  mount('/capabilities')
  expect($routeCollapsedPanes.get().has('sessions')).toBe(true)
  // The pane is collapsed for the route, not hidden: it stays mounted (route-collapse.test.tsx).
  expect($hiddenTreePanes.get().has('sessions')).toBe(false)
  act(() => setTreePaneHidden('sessions', false))
  expect($routeCollapsedPanes.get().has('sessions')).toBe(true)
  expect($hiddenStripTabs.get().has('sessions')).toBe(false)
  fireEvent.click(screen.getByRole('button', { name: 'Chats' }))
  expect($routeCollapsedPanes.get().has('sessions')).toBe(false)
  expect($sidebarOpen.get()).toBe(true)
})

// Each shell keeps its own arrangement (lib/layout-persistence.ts): a tab hidden in classic is
// classic's, the redesign never inherits it, and switching back hands it back untouched.
it('restores classic page visibility when v2 is disabled while preserving explicit tab hides', () => {
  mount('/capabilities')
  act(() => $shellChoice.set(false))
  expect(screen.queryByRole('navigation')).toBeNull()
  expect($routeCollapsedPanes.get().has('sessions')).toBe(false)
  expect($hiddenTreePanes.get().has('sessions')).toBe(false)
  act(() => $hiddenStripTabs.set(new Set(['sessions'])))
  act(() => $shellChoice.set(true))
  fireEvent.click(screen.getByRole('button', { name: 'Chats' }))
  expect($hiddenTreePanes.get().has('sessions')).toBe(false)
  act(() => $shellChoice.set(false))
  expect($hiddenStripTabs.get().has('sessions')).toBe(true)
  expect($hiddenTreePanes.get().has('sessions')).toBe(true)
})

it('shows sessions for a focused chat tile even with a global page in the workspace', () => {
  $layoutTree.set(
    split('row', [
      group(['workspace'], { id: 'workspace-group' }),
      group(['session-tile:chat-one'], { id: 'chat-group' })
    ])
  )
  noteActiveTreeGroup('workspace-group')
  mount('/capabilities')
  expect($routeCollapsedPanes.get().has('sessions')).toBe(true)
  act(() => noteActiveTreeGroup('chat-group'))
  expect($routeCollapsedPanes.get().has('sessions')).toBe(false)
  act(() => noteActiveTreeGroup('workspace-group'))
  expect($routeCollapsedPanes.get().has('sessions')).toBe(true)
})
