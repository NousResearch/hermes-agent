import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, expect, it, vi } from 'vitest'

import { group, split } from '@/components/pane-shell/tree/model'
import { $layoutTree, noteActiveTreeGroup } from '@/components/pane-shell/tree/store'
import { RootTooltipProvider } from '@/components/ui/tooltip'

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
  ['Agent settings', '/profiles'],
  ['Skills', '/capabilities?tab=skills'],
  ['Library', '/artifacts'],
  ['Routines', '/cron']
])('opens %s using its existing destination', (label, route) => {
  const navigate = mount()
  fireEvent.click(screen.getByRole('button', { name: label }))
  expect(navigate).toHaveBeenCalledWith(expect.objectContaining({ route }))
})

it('opens Plugins through Capabilities and identifies the active section', () => {
  const navigate = mount('/capabilities?tab=plugins')
  const button = screen.getByRole('button', { name: 'Capabilities' })
  expect(button.getAttribute('aria-current')).toBe('page')
  expect(screen.getByRole('button', { name: 'Skills' }).hasAttribute('aria-current')).toBe(false)
  fireEvent.keyDown(button, { key: 'Enter' })
  fireEvent.click(screen.getByRole('menuitem', { name: 'Plugins' }))
  expect(navigate).toHaveBeenCalledWith(expect.objectContaining({ route: '/capabilities?tab=plugins' }))
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
  const skills = screen.getByRole('button', { name: 'Skills' })
  const chats = screen.getByRole('button', { name: 'Chats' })
  expect(skills.getAttribute('aria-current')).toBe('page')
  act(() => noteActiveTreeGroup('chat-group'))
  expect(chats.getAttribute('aria-current')).toBe('page')
  expect(skills.hasAttribute('aria-current')).toBe(false)
  act(() => noteActiveTreeGroup('workspace-group'))
  expect(skills.getAttribute('aria-current')).toBe('page')
  expect(chats.hasAttribute('aria-current')).toBe(false)
})
