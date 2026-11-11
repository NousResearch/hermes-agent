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
  $layoutTree.set(null)
  noteActiveTreeGroup(null)
  setSidebarOpen(true)
})

afterEach(() => {
  cleanup()
  $shellChoice.set(false)
  $hiddenStripTabs.set(new Set())
  $hiddenTreePanes.set(new Set())
  $layoutTree.set(null)
  noteActiveTreeGroup(null)
  setSidebarOpen(true)
})

it('keeps global navigation while the sidebar is closed and preserves that choice across pages', () => {
  mount()
  act(() => setSidebarOpen(false))
  expect(screen.getByRole('navigation', { name: 'Hermes' })).toBeTruthy()
  fireEvent.click(screen.getByRole('button', { name: 'Capabilities' }))
  expect($hiddenTreePanes.get().has('sessions')).toBe(true)
  expect($sidebarOpen.get()).toBe(false)
  fireEvent.click(screen.getByRole('button', { name: 'Chats' }))
  expect($hiddenTreePanes.get().has('sessions')).toBe(false)
  expect($sidebarOpen.get()).toBe(false)
})

it('hides sessions on global pages, including after an attempted pane reveal, without saving a hidden tab', () => {
  mount('/capabilities')
  expect($hiddenTreePanes.get().has('sessions')).toBe(true)
  act(() => setTreePaneHidden('sessions', false))
  expect($hiddenTreePanes.get().has('sessions')).toBe(true)
  expect($hiddenStripTabs.get().has('sessions')).toBe(false)
  fireEvent.click(screen.getByRole('button', { name: 'Chats' }))
  expect($hiddenTreePanes.get().has('sessions')).toBe(false)
  expect($sidebarOpen.get()).toBe(true)
})

it('restores classic page visibility when v2 is disabled while preserving explicit tab hides', () => {
  mount('/capabilities')
  act(() => $shellChoice.set(false))
  expect(screen.queryByRole('navigation')).toBeNull()
  expect($hiddenTreePanes.get().has('sessions')).toBe(false)
  act(() => {
    $hiddenStripTabs.set(new Set(['sessions']))
    $shellChoice.set(true)
  })
  fireEvent.click(screen.getByRole('button', { name: 'Chats' }))
  expect($hiddenTreePanes.get().has('sessions')).toBe(true)
  act(() => $shellChoice.set(false))
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
  expect($hiddenTreePanes.get().has('sessions')).toBe(true)
  act(() => noteActiveTreeGroup('chat-group'))
  expect($hiddenTreePanes.get().has('sessions')).toBe(false)
  act(() => noteActiveTreeGroup('workspace-group'))
  expect($hiddenTreePanes.get().has('sessions')).toBe(true)
})
