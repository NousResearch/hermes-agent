import { act, cleanup, fireEvent, render, screen, within } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, expect, it, vi } from 'vitest'

import { group, split } from '@/components/pane-shell/tree/model'
import { $layoutTree, noteActiveTreeGroup } from '@/components/pane-shell/tree/store'
import { RootTooltipProvider } from '@/components/ui/tooltip'
import { registry } from '@/contrib/registry'
import { en } from '@/i18n/en'
import { $interfaceMode } from '@/store/interface-mode'
import { SIDEBAR_NAV_PREFS_AREA } from '@/store/sidebar-nav'

import { ROUTES_AREA, SIDEBAR_NAV_AREA } from '../../routes'

import { SimplifiedGlobalNav } from './simplified-global-nav'

afterEach(() => {
  cleanup()
  $layoutTree.set(null)
  noteActiveTreeGroup(null)
})

// Each destination is picked from another one: the open destination's button folds its panel instead.
it.each([
  [en.simplifiedAgents.chats, '/', '/artifacts'],
  [en.profiles.title, '/profiles', '/'],
  [en.sidebar.nav.capabilities, '/capabilities', '/'],
  [en.simplifiedAgents.library, '/artifacts', '/'],
  [en.cron.title, '/cron', '/']
])('navigates from the core %s destination to %s without plugin pages', (label, route, from) => {
  const onNavigate = vi.fn()
  render(
    <MemoryRouter initialEntries={[from]}>
      <RootTooltipProvider>
        <SimplifiedGlobalNav onNavigate={onNavigate} />
      </RootTooltipProvider>
    </MemoryRouter>
  )
  fireEvent.click(screen.getByRole('button', { name: label }))
  expect(onNavigate).toHaveBeenCalledExactlyOnceWith(expect.objectContaining({ route }))
})

it('keeps plugin pages reachable and respects visibility, preferences, and unloading', () => {
  const previousMode = $interfaceMode.get()
  $interfaceMode.set('simple')

  const dispose = registry.registerMany([
    { area: ROUTES_AREA, id: 'test-board-route', data: { path: '/board' }, render: () => null },
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
    expect(capabilities.hasAttribute('aria-current')).toBe(false)
    expect(screen.queryByRole('button', { name: 'Advanced page' })).toBeNull()
    expect(screen.queryByRole('button', { name: 'Hidden page' })).toBeNull()
    expect(screen.queryByRole('button', { name: 'Invalid page' })).toBeNull()
    const board = screen.getByRole('button', { name: 'Test board' })
    expect(board.getAttribute('aria-current')).toBe('page')
    expect(screen.getByRole('button', { name: 'Chats' }).hasAttribute('aria-current')).toBe(false)
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
    act(dispose)
    expect(screen.queryByRole('button', { name: 'Test board' })).toBeNull()
  } finally {
    cleanup()
    dispose()
    $interfaceMode.set(previousMode)
  }
})

it('opens the contributing plugin settings from its page icon without navigating to the page', () => {
  const dispose = registry.registerMany([
    { area: SIDEBAR_NAV_AREA, id: 'custom:gantt', source: 'plugin:custom', data: { label: 'Gantt', path: '/gantt' } },
    {
      area: 'settings.plugins',
      id: 'custom:preferences',
      source: 'plugin:custom',
      title: 'Preferences',
      render: () => null
    }
  ])

  try {
    const onNavigate = vi.fn()
    render(
      <MemoryRouter>
        <RootTooltipProvider>
          <SimplifiedGlobalNav onNavigate={onNavigate} />
        </RootTooltipProvider>
      </MemoryRouter>
    )
    fireEvent.contextMenu(screen.getByRole('button', { name: 'Gantt' }))
    expect(onNavigate).not.toHaveBeenCalled()
    fireEvent.click(screen.getByRole('menuitem'))
    expect(onNavigate).toHaveBeenCalledWith(
      expect.objectContaining({ route: '/settings?tab=plugins&plugin=custom&profile=default' })
    )
  } finally {
    cleanup()
    dispose()
  }
})

it('does not offer a broken settings shortcut for a plugin without a settings page', () => {
  const dispose = registry.register({
    area: SIDEBAR_NAV_AREA,
    id: 'plain:page',
    source: 'plugin:plain',
    data: { label: 'Plain', path: '/plain' }
  })

  try {
    render(
      <MemoryRouter>
        <RootTooltipProvider>
          <SimplifiedGlobalNav onNavigate={vi.fn()} />
        </RootTooltipProvider>
      </MemoryRouter>
    )
    fireEvent.contextMenu(screen.getByRole('button', { name: 'Plain' }))
    expect(screen.queryByRole('menu')).toBeNull()
  } finally {
    cleanup()
    dispose()
  }
})

const railLabels = () =>
  within(screen.getByRole('navigation'))
    .getAllByRole('button')
    .map(button => button.getAttribute('aria-label'))

it('applies nav preferences to the rail’s core destinations as classic does to its nav rows', () => {
  const dispose = registry.registerMany([
    { area: SIDEBAR_NAV_AREA, id: 'mgr:board', source: 'plugin:mgr', data: { label: 'Board', path: '/board' } },
    {
      area: SIDEBAR_NAV_PREFS_AREA,
      id: 'mgr:prefs',
      source: 'plugin:mgr',
      // `chat`, `profiles` and `settings` name no movable row: inert, like ids classic has no row for.
      data: { hide: ['cron', 'capabilities', 'chat', 'profiles', 'settings'], order: ['mgr:board', 'artifacts'] }
    }
  ])

  try {
    render(
      <MemoryRouter>
        <RootTooltipProvider>
          <SimplifiedGlobalNav onNavigate={vi.fn()} />
        </RootTooltipProvider>
      </MemoryRouter>
    )

    expect(railLabels()).toEqual([
      en.simplifiedAgents.chats,
      en.profiles.title,
      'Board',
      en.simplifiedAgents.library,
      // Capabilities hosts the Plugins tab, the way back to a plugin's off switch: never hidden.
      en.sidebar.nav.capabilities,
      en.commandCenter.settings
    ])
    act(dispose)
    expect(railLabels()).toEqual([
      en.simplifiedAgents.chats,
      en.profiles.title,
      en.sidebar.nav.capabilities,
      en.simplifiedAgents.library,
      en.cron.title,
      en.commandCenter.settings
    ])
  } finally {
    cleanup()
    dispose()
  }
})

it('marks a plugin destination open on its nested routes, not on a sibling prefix', () => {
  const dispose = registry.register({
    area: SIDEBAR_NAV_AREA,
    id: 'nested:board',
    source: 'plugin:nested',
    data: { label: 'Board', path: '/board' }
  })

  try {
    for (const [path, open] of [
      ['/board/cards/42', true],
      ['/boardroom', false]
    ] as const) {
      render(
        <MemoryRouter initialEntries={[path]}>
          <RootTooltipProvider>
            <SimplifiedGlobalNav onNavigate={vi.fn()} />
          </RootTooltipProvider>
        </MemoryRouter>
      )
      expect(screen.getByRole('button', { name: 'Board' }).getAttribute('aria-current')).toBe(open ? 'page' : null)
      cleanup()
    }
  } finally {
    dispose()
  }
})
