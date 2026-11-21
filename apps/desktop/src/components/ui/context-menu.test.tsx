import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest'

import {
  ContextMenu,
  ContextMenuContent,
  ContextMenuItem,
  ContextMenuSub,
  ContextMenuSubContent,
  ContextMenuSubTrigger,
  ContextMenuTrigger
} from './context-menu'

// Radix menus use pointer capture and scrollIntoView; jsdom has neither.
beforeAll(() => {
  Element.prototype.hasPointerCapture ??= () => false
  Element.prototype.releasePointerCapture ??= () => undefined
  Element.prototype.scrollIntoView ??= () => undefined
})

afterEach(cleanup)

// iPadOS Safari sends pointer and touch events for an Apple Pencil tap, but no
// mousedown/mouseup/click (#150605). Radix menu rows act only on click.
// jsdom lays rows out at 0×0, so (0, 0) is on the row and anything else is off it.
function pencilTap(element: Element, end = { clientX: 0, clientY: 0 }) {
  const start = { clientX: 0, clientY: 0, identifier: 1, target: element }

  fireEvent.pointerDown(element, { button: 0, pointerType: 'pen' })
  fireEvent.touchStart(element, { changedTouches: [start], touches: [start] })
  fireEvent.pointerUp(element, { button: 0, pointerType: 'pen' })

  // false = the touchend was cancelled, so Safari sends no compatibility click.
  return fireEvent.touchEnd(element, { changedTouches: [{ ...start, ...end }], touches: [] })
}

function openMenu(onSelect?: (event: Event) => void) {
  render(
    <ContextMenu>
      <ContextMenuTrigger>Session</ContextMenuTrigger>
      <ContextMenuContent>
        <ContextMenuItem onSelect={onSelect}>Rename</ContextMenuItem>
        <ContextMenuSub>
          <ContextMenuSubTrigger>Move to project</ContextMenuSubTrigger>
          <ContextMenuSubContent>
            <ContextMenuItem>Inbox</ContextMenuItem>
          </ContextMenuSubContent>
        </ContextMenuSub>
      </ContextMenuContent>
    </ContextMenu>
  )

  fireEvent.contextMenu(screen.getByText('Session'))
}

describe('Apple Pencil taps on iPadOS Safari (#150605)', () => {
  it('opens a context submenu from a pencil tap that sends no click', () => {
    openMenu()

    const cancelled = !pencilTap(screen.getByRole('menuitem', { name: 'Move to project' }))

    expect(screen.queryByRole('menuitem', { name: 'Inbox' })).not.toBeNull()
    expect(cancelled).toBe(true)
  })

  it('selects a context item once from a pencil tap', () => {
    const onSelect = vi.fn((event: Event) => event.preventDefault())

    openMenu(onSelect)

    const cancelled = !pencilTap(screen.getByRole('menuitem', { name: 'Rename' }))

    expect(onSelect).toHaveBeenCalledTimes(1)
    expect(cancelled).toBe(true)
  })
})
