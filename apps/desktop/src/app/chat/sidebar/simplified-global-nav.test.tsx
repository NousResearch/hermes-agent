import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, expect, it, vi } from 'vitest'

import { group, split } from '@/components/pane-shell/tree/model'
import { $layoutTree, noteActiveTreeGroup } from '@/components/pane-shell/tree/store'
import { RootTooltipProvider } from '@/components/ui/tooltip'
import { en } from '@/i18n/en'

import { SimplifiedGlobalNav } from './simplified-global-nav'

afterEach(() => {
  cleanup()
  $layoutTree.set(null)
  noteActiveTreeGroup(null)
})

function mount(route = '/') {
  const navigate = vi.fn()
  render(
    <MemoryRouter initialEntries={[route]}>
      <RootTooltipProvider>
        <SimplifiedGlobalNav onNavigate={navigate} />
      </RootTooltipProvider>
    </MemoryRouter>
  )

  return navigate
}

it.each([
  [en.profiles.title, '/profiles'],
  [en.sidebar.nav.capabilities, '/capabilities'],
  [en.simplifiedAgents.library, '/artifacts'],
  [en.cron.title, '/cron']
])('opens %s using its existing destination', (label, route) => {
  const navigate = mount()
  fireEvent.click(screen.getByRole('button', { name: label }))
  expect(navigate).toHaveBeenCalledWith(expect.objectContaining({ route }))
})

it.each(['skills', 'plugins', 'connectors', 'toolsets'])(
  'keeps Capabilities active for its %s tab without a dropdown',
  tab => {
    const navigate = mount(`/capabilities?tab=${tab}`)
    const button = screen.getByRole('button', { name: 'Capabilities' })
    expect(button.getAttribute('aria-current')).toBe('page')
    expect(screen.queryByRole('button', { name: 'Skills' })).toBeNull()
    fireEvent.click(button)
    expect(navigate).toHaveBeenCalledWith(expect.objectContaining({ route: '/capabilities' }))
    expect(screen.queryByRole('menu')).toBeNull()
  }
)

it.each([
  ['/settings?agentProfile=research&panel=settings&tab=config:model', en.profiles.title],
  ['/settings?tab=config:appearance', 'Settings']
])('marks the owning rail destination on %s', (route, label) => {
  mount(route)
  expect(screen.getByRole('button', { current: 'page' }).getAttribute('aria-label')).toBe(label)
})

it('opens account settings without inventing a global profile or changing the active agent', () => {
  const navigate = mount()
  fireEvent.keyDown(screen.getByRole('button', { name: 'Settings' }), { key: 'Enter' })
  fireEvent.click(screen.getByRole('menuitem', { name: 'Accounts' }))
  expect(navigate).toHaveBeenCalledWith(expect.objectContaining({ route: '/settings?tab=providers&pview=accounts' }))
})

it('keeps existing settings sections reachable from footer submenus', async () => {
  const navigate = mount()
  fireEvent.keyDown(screen.getByRole('button', { name: 'Settings' }), { key: 'Enter' })
  fireEvent.keyDown(screen.getByRole('menuitem', { name: 'Appearance' }), { key: 'ArrowRight' })
  fireEvent.click(await screen.findByRole('menuitem', { name: 'Notifications' }))
  expect(navigate).toHaveBeenCalledWith(expect.objectContaining({ route: '/settings?tab=notifications' }))
})

it('follows the focused chat tile instead of the workspace page underneath it', () => {
  $layoutTree.set(
    split('row', [
      group(['workspace'], { id: 'workspace-group' }),
      group(['session-tile:chat-one'], { id: 'chat-group' })
    ])
  )
  noteActiveTreeGroup('workspace-group')
  mount('/capabilities?tab=skills')
  const capabilities = screen.getByRole('button', { name: 'Capabilities' })
  const chats = screen.getByRole('button', { name: 'Chats' })
  expect(capabilities.getAttribute('aria-current')).toBe('page')
  act(() => noteActiveTreeGroup('chat-group'))
  expect(chats.getAttribute('aria-current')).toBe('page')
  expect(capabilities.hasAttribute('aria-current')).toBe(false)
  act(() => noteActiveTreeGroup('workspace-group'))
  expect(capabilities.getAttribute('aria-current')).toBe('page')
  expect(chats.hasAttribute('aria-current')).toBe(false)
})
