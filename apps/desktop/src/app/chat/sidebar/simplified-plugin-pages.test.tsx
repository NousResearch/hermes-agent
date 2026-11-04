import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, expect, it, vi } from 'vitest'

import { group, split } from '@/components/pane-shell/tree/model'
import { $layoutTree, noteActiveTreeGroup } from '@/components/pane-shell/tree/store'
import { RootTooltipProvider } from '@/components/ui/tooltip'
import { registry } from '@/contrib/registry'
import { $interfaceMode } from '@/store/interface-mode'
import { SIDEBAR_NAV_PREFS_AREA } from '@/store/sidebar-nav'

import { SIDEBAR_NAV_AREA } from '../../routes'

import { SimplifiedGlobalNav } from './simplified-global-nav'

afterEach(() => {
  cleanup()
  $layoutTree.set(null)
  noteActiveTreeGroup(null)
})

it('omits the plugin divider when no pages are enabled', () => {
  render(
    <MemoryRouter>
      <RootTooltipProvider>
        <SimplifiedGlobalNav onNavigate={vi.fn()} />
      </RootTooltipProvider>
    </MemoryRouter>
  )
  fireEvent.keyDown(screen.getByRole('button', { name: 'Capabilities' }), { key: 'Enter' })
  expect(screen.getAllByRole('menuitem')).toHaveLength(4)
  expect(screen.queryByRole('separator')).toBeNull()
})

it('keeps plugin pages reachable and respects visibility, preferences, and unloading', () => {
  const previousMode = $interfaceMode.get()
  $interfaceMode.set('simple')

  const dispose = registry.registerMany([
    {
      area: SIDEBAR_NAV_AREA,
      id: 'test-board',
      source: 'plugin:test-plugin',
      data: { label: 'Test board', path: '/board' }
    },
    {
      area: SIDEBAR_NAV_AREA,
      id: 'test-advanced',
      data: { label: 'Advanced page', path: '/advanced', tier: 'advanced' }
    },
    { area: SIDEBAR_NAV_AREA, id: 'test-hidden', data: { label: 'Hidden page', path: '/hidden' } },
    { area: SIDEBAR_NAV_AREA, id: 'test-invalid', data: { label: 'Invalid page', path: 'invalid' } },
    { area: SIDEBAR_NAV_PREFS_AREA, id: 'test-prefs', data: { hide: ['test-hidden'] } }
  ])

  try {
    const onNavigate = vi.fn()
    render(
      <MemoryRouter initialEntries={['/board']}>
        <RootTooltipProvider>
          <SimplifiedGlobalNav onNavigate={onNavigate} />
        </RootTooltipProvider>
      </MemoryRouter>
    )
    const capabilities = screen.getByRole('button', { name: 'Capabilities' })
    expect(capabilities.getAttribute('aria-current')).toBe('page')
    fireEvent.keyDown(capabilities, { key: 'Enter' })
    expect(screen.getByRole('separator')).toBeTruthy()
    expect(screen.getAllByRole('menuitem').map(item => item.textContent)).toEqual([
      'Connectors',
      'Plugins',
      'Tools',
      'Maintenance',
      'Test board'
    ])
    expect(screen.queryByRole('menuitem', { name: 'Advanced page' })).toBeNull()
    expect(screen.queryByRole('menuitem', { name: 'Hidden page' })).toBeNull()
    expect(screen.queryByRole('menuitem', { name: 'Invalid page' })).toBeNull()
    const board = screen.getByRole('menuitem', { name: 'Test board' })
    expect(board.getAttribute('aria-current')).toBe('page')
    act(() => {
      $layoutTree.set(
        split('row', [
          group(['workspace'], { id: 'workspace-group' }),
          group(['session-tile:chat-one'], { id: 'chat-group' })
        ])
      )
      noteActiveTreeGroup('chat-group')
    })
    expect(board.hasAttribute('aria-current')).toBe(false)
    act(() => noteActiveTreeGroup('workspace-group'))
    expect(board.getAttribute('aria-current')).toBe('page')
    fireEvent.click(board)
    expect(onNavigate).toHaveBeenCalledWith(expect.objectContaining({ route: '/board' }))
    fireEvent.keyDown(capabilities, { key: 'Enter' })
    act(dispose)
    expect(screen.queryByRole('menuitem', { name: 'Test board' })).toBeNull()
    expect(screen.queryByRole('separator')).toBeNull()
  } finally {
    cleanup()
    dispose()
    $interfaceMode.set(previousMode)
  }
})
