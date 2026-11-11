import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { Wrench } from '@/lib/icons'
import { $shellChoice } from '@/store/shell-variant'

import { OverlayBreadcrumbHeader } from './overlay-breadcrumb-header'

afterEach(() => {
  cleanup()
  $shellChoice.set(null)
})

it('shares the V2 destination title while retaining breadcrumb and toolbar actions in classic', () => {
  $shellChoice.set(true)
  const openGroup = vi.fn()
  const openChild = vi.fn()
  const refresh = vi.fn()
  const group = { id: 'plugins', label: 'Plugins', active: true, icon: Wrench, onSelect: openGroup }
  const child = { ...group, id: 'kanban', label: 'Kanban', onSelect: openChild }
  const grandchild = { ...child, id: 'board', label: 'Board' }
  const { container } = render(
    <OverlayBreadcrumbHeader
      child={child}
      grandchild={grandchild}
      group={group}
      rootLabel="Settings"
      trailing={<button onClick={refresh}>Refresh</button>}
    />
  )
  expect(screen.getByRole('heading', { level: 1, name: 'Board' })).toBeTruthy()
  expect(container.querySelector('[aria-current="page"]')?.textContent).toBe('Board')
  fireEvent.click(screen.getByRole('button', { name: 'Plugins' }))
  fireEvent.click(screen.getByRole('button', { name: 'Kanban' }))
  expect(openGroup).toHaveBeenCalledOnce()
  expect(openChild).toHaveBeenCalledOnce()
  fireEvent.click(screen.getByRole('button', { name: 'Refresh' }))

  act(() => $shellChoice.set(false))
  expect(screen.queryByRole('heading')).toBeNull()
  fireEvent.click(screen.getByRole('button', { name: 'Refresh' }))
  expect(refresh).toHaveBeenCalledTimes(2)
})
