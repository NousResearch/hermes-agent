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

// The context wrapper covers a whole pane body, so an editable inside it must be
// able to keep the gesture: radix's trigger unconditionally preventDefaults and
// opens, which would swallow the app-level menu that carries the clipboard verbs.
it('renders a disabled context wrapper bare so an editable keeps the gesture', async () => {
  const items = (kit: MenuKit) => <kit.Item>Zone action</kit.Item>
  const { container, rerender } = render(
    <ActionsContextMenu disabled items={items}>
      <div contentEditable data-testid="field" />
    </ActionsContextMenu>
  )

  const field = screen.getByTestId('field')

  expect(field.closest('[data-slot="context-menu-trigger"]')).toBeNull()

  let bubbled = 0

  container.addEventListener('contextmenu', () => (bubbled += 1))
  fireEvent.contextMenu(field, { clientX: 10, clientY: 10 })

  expect(bubbled).toBe(1)
  expect(screen.queryByRole('menu')).toBeNull()

  rerender(
    <ActionsContextMenu items={items}>
      <div contentEditable data-testid="field" />
    </ActionsContextMenu>
  )

  fireEvent.contextMenu(screen.getByTestId('field'), { clientX: 10, clientY: 10 })
  expect(await screen.findByRole('menuitem', { name: 'Zone action' })).toBeTruthy()
})
