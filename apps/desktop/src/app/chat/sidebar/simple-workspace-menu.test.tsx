import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { $layoutEditMode } from '@/components/pane-shell/edit-mode'
import { resetBinding, setBinding } from '@/store/keybinds'

import { SimpleWorkspaceMenu } from './simple-workspace-menu'

afterEach(() => {
  cleanup()
  $layoutEditMode.set(false)
  resetBinding('nav.messaging')
})

it('names the current secondary destination, marks it, and preserves its navigation', () => {
  setBinding('nav.messaging', ['mod+shift+m'])
  const navigate = vi.fn()
  const icon = () => null

  const items = [
    { id: 'messaging', route: '/messaging', label: 'Messaging', icon, keybindActionId: 'nav.messaging' },
    { id: 'kanban', route: '/kanban', label: 'Kanban', icon }
  ]

  render(<SimpleWorkspaceMenu activeRoute="/kanban" items={items} onNavigate={navigate} />)
  fireEvent.pointerDown(screen.getByRole('button', { name: 'More · Kanban' }), {
    button: 0,
    ctrlKey: false,
    pointerType: 'mouse'
  })
  expect(screen.getByRole('menuitem', { name: 'Kanban' }).getAttribute('aria-current')).toBe('page')
  expect(screen.getByRole('menuitem', { name: /^Messaging/ }).querySelector('kbd')).not.toBeNull()
  fireEvent.click(screen.getByRole('menuitem', { name: 'Kanban' }))
  expect(navigate).toHaveBeenCalledWith(items[1])
})

it('opens Layout from the bottom controls through the shared editor action', () => {
  render(<SimpleWorkspaceMenu activeRoute="/" items={[]} onNavigate={vi.fn()} />)
  fireEvent.click(screen.getByRole('button', { name: 'Layout editor' }))
  expect($layoutEditMode.get()).toBe(true)
})
