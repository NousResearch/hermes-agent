import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { ActionsContextMenu, ActionsMenu, type MenuKit } from './actions-menu'

afterEach(cleanup)

it.each(['dropdown', 'context'] as const)('defers %s items until opened and reads the latest actions', async kind => {
  const selected = vi.fn()
  const items = vi.fn((kit: MenuKit) => <kit.Item onSelect={selected}>Original action</kit.Item>)
  const nextItems = vi.fn((kit: MenuKit) => <kit.Item onSelect={selected}>Latest action</kit.Item>)
  const Menu = kind === 'dropdown' ? ActionsMenu : ActionsContextMenu

  const view = (renderItems: typeof items) => (
    <Menu items={renderItems}>
      <button type="button">Actions</button>
    </Menu>
  )

  const { rerender } = render(view(items))
  expect(items).not.toHaveBeenCalled()
  rerender(view(nextItems))
  expect(nextItems).not.toHaveBeenCalled()

  const trigger = screen.getByRole('button', { name: 'Actions' })

  if (kind === 'dropdown') {
    fireEvent.pointerDown(trigger, { button: 0, pointerType: 'mouse' })
  } else {
    fireEvent.contextMenu(trigger, { clientX: 20, clientY: 20 })
  }

  fireEvent.click(await screen.findByRole('menuitem', { name: 'Latest action' }))
  expect(selected).toHaveBeenCalledTimes(1)
  await waitFor(() => expect(screen.queryByRole('menu')).toBeNull())
  const callsAfterClose = nextItems.mock.calls.length
  rerender(view(nextItems))
  expect(nextItems).toHaveBeenCalledTimes(callsAfterClose)
})
