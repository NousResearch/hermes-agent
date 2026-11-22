import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { useState } from 'react'
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest'

import { Dialog, DialogContent } from './dialog'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuSearch,
  DropdownMenuSub,
  DropdownMenuSubContent,
  DropdownMenuSubTrigger
} from './dropdown-menu'

// Radix menus use pointer capture and scrollIntoView; jsdom has neither.
beforeAll(() => {
  Element.prototype.hasPointerCapture ??= () => false
  Element.prototype.releasePointerCapture ??= () => undefined
  Element.prototype.scrollIntoView ??= () => undefined
})

afterEach(() => {
  cleanup()
  vi.useRealTimers()
})

const hover = (element: Element) => fireEvent.pointerMove(element, { pointerType: 'mouse' })
const unhover = (element: Element) => fireEvent.pointerLeave(element, { pointerType: 'mouse' })

function SearchableMenu({ withSearch = true }: { withSearch?: boolean }) {
  return (
    <DropdownMenu open>
      <DropdownMenuContent>
        {withSearch && <DropdownMenuSearch aria-label="Search" />}
        <DropdownMenuItem>Plain row</DropdownMenuItem>
        <DropdownMenuRadioGroup value="a">
          <DropdownMenuRadioItem value="a">Radio row</DropdownMenuRadioItem>
        </DropdownMenuRadioGroup>
        <DropdownMenuSub>
          <DropdownMenuSubTrigger>Sub row</DropdownMenuSubTrigger>
          <DropdownMenuSubContent>
            <DropdownMenuItem>Sub option</DropdownMenuItem>
          </DropdownMenuSubContent>
        </DropdownMenuSub>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

describe('DropdownMenuSearch hover focus', () => {
  it('keeps focus in the search field while the mouse moves over rows (#53980)', () => {
    render(<SearchableMenu />)

    const search = screen.getByRole('textbox', { name: 'Search' })
    search.focus()
    expect(search.ownerDocument.activeElement).toBe(search)

    for (const name of ['Plain row', 'Radio row', 'Sub row']) {
      const row = screen.getByText(name).closest('[role^="menuitem"]')!
      hover(row)
      expect(search.ownerDocument.activeElement).toBe(search)
      unhover(row)
      expect(search.ownerDocument.activeElement).toBe(search)
    }
  })

  it('still highlights the hovered row without taking focus', () => {
    render(<SearchableMenu />)

    screen.getByRole('textbox', { name: 'Search' }).focus()
    const row = screen.getByRole('menuitem', { name: 'Plain row' })

    hover(row)
    expect(row.hasAttribute('data-highlighted')).toBe(true)

    unhover(row)
    expect(row.hasAttribute('data-highlighted')).toBe(false)
  })

  it('opens a hovered submenu and closes it when another row is hovered, without taking focus', () => {
    vi.useFakeTimers()
    render(<SearchableMenu />)

    const search = screen.getByRole('textbox', { name: 'Search' })
    search.focus()

    hover(screen.getByRole('menuitem', { name: 'Sub row' }))
    act(() => vi.advanceTimersByTime(200))

    expect(screen.queryByRole('menuitem', { name: 'Sub option' })).not.toBeNull()
    expect(search.ownerDocument.activeElement).toBe(search)

    hover(screen.getByRole('menuitem', { name: 'Plain row' }))
    act(() => vi.advanceTimersByTime(500))

    expect(screen.queryByRole('menuitem', { name: 'Sub option' })).toBeNull()
    expect(search.ownerDocument.activeElement).toBe(search)
  })

  it('drops a pending submenu hover-open once the user types, so the submenu cannot take the caret', () => {
    vi.useFakeTimers()
    render(<SearchableMenu />)

    const search = screen.getByRole('textbox', { name: 'Search' })
    search.focus()

    hover(screen.getByRole('menuitem', { name: 'Sub row' }))
    fireEvent.keyDown(search, { key: 'g' })
    act(() => vi.advanceTimersByTime(200))

    expect(screen.queryByRole('menuitem', { name: 'Sub option' })).toBeNull()
    expect(search.ownerDocument.activeElement).toBe(search)
  })

  it('keeps Radix hover-to-focus once focus has left the search field', () => {
    render(<SearchableMenu />)

    const row = screen.getByRole('menuitem', { name: 'Plain row' })
    screen.getByRole('menuitemradio', { name: 'Radio row' }).focus()

    hover(row)
    expect(row.ownerDocument.activeElement).toBe(row)
  })

  it('leaves menus without a search field on Radix hover-to-focus', () => {
    render(<SearchableMenu withSearch={false} />)

    const row = screen.getByRole('menuitem', { name: 'Plain row' })

    hover(row)
    expect(row.ownerDocument.activeElement).toBe(row)
  })
})

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

describe('Apple Pencil taps on iPadOS Safari (#150605)', () => {
  function PencilMenu({ onSelect }: { onSelect?: (event: Event) => void }) {
    return (
      <DropdownMenu open>
        <DropdownMenuContent>
          <DropdownMenuItem onSelect={onSelect}>Rename</DropdownMenuItem>
          <DropdownMenuSub>
            <DropdownMenuSubTrigger>Appearance</DropdownMenuSubTrigger>
            <DropdownMenuSubContent>
              <DropdownMenuItem>Pin</DropdownMenuItem>
            </DropdownMenuSubContent>
          </DropdownMenuSub>
        </DropdownMenuContent>
      </DropdownMenu>
    )
  }

  it('opens a submenu from a pencil tap that sends no click', () => {
    render(<PencilMenu />)

    const cancelled = !pencilTap(screen.getByRole('menuitem', { name: 'Appearance' }))

    expect(screen.queryByRole('menuitem', { name: 'Pin' })).not.toBeNull()
    expect(cancelled).toBe(true)
  })

  it('selects an item once from a pencil tap and cancels the native click', () => {
    const onSelect = vi.fn((event: Event) => event.preventDefault())

    render(<PencilMenu onSelect={onSelect} />)

    const cancelled = !pencilTap(screen.getByRole('menuitem', { name: 'Rename' }))

    expect(onSelect).toHaveBeenCalledTimes(1)
    expect(cancelled).toBe(true)
  })

  it('does nothing when the pencil is lifted off the row', () => {
    const onSelect = vi.fn((event: Event) => event.preventDefault())

    render(<PencilMenu onSelect={onSelect} />)

    expect(pencilTap(screen.getByRole('menuitem', { name: 'Rename' }), { clientX: 40, clientY: 40 })).toBe(true)
    expect(onSelect).not.toHaveBeenCalled()
  })

  it('leaves finger taps to the browser click', () => {
    const onSelect = vi.fn((event: Event) => event.preventDefault())

    render(<PencilMenu onSelect={onSelect} />)

    const row = screen.getByRole('menuitem', { name: 'Rename' })

    fireEvent.pointerDown(row, { button: 0, pointerType: 'touch' })
    fireEvent.pointerUp(row, { button: 0, pointerType: 'touch' })
    expect(fireEvent.touchEnd(row, { changedTouches: [{ clientX: 0, clientY: 0, identifier: 1, target: row }] })).toBe(
      true
    )
    expect(onSelect).not.toHaveBeenCalled()

    fireEvent.click(row)
    expect(onSelect).toHaveBeenCalledTimes(1)
  })

  it('leaves pens that send a click and no touch events alone', () => {
    const onSelect = vi.fn((event: Event) => event.preventDefault())

    render(<PencilMenu onSelect={onSelect} />)

    const row = screen.getByRole('menuitem', { name: 'Rename' })

    fireEvent.pointerDown(row, { button: 0, pointerType: 'pen' })
    fireEvent.pointerUp(row, { button: 0, pointerType: 'pen' })
    fireEvent.click(row)

    expect(onSelect).toHaveBeenCalledTimes(1)
  })
})

describe('DropdownMenuSubContent portal', () => {
  function OpenSubmenu({ portalContainer }: { portalContainer?: HTMLElement | null }) {
    return (
      <DropdownMenu open>
        <DropdownMenuContent portalContainer={portalContainer}>
          <DropdownMenuSub open>
            <DropdownMenuSubTrigger>Model</DropdownMenuSubTrigger>
            <DropdownMenuSubContent>
              <DropdownMenuItem>High</DropdownMenuItem>
            </DropdownMenuSubContent>
          </DropdownMenuSub>
        </DropdownMenuContent>
      </DropdownMenu>
    )
  }

  it('keeps a submenu on the body portal, at z-50, outside a dialog', () => {
    render(<OpenSubmenu />)

    const sub = screen.getByText('High').closest('[data-slot="dropdown-menu-sub-content"]')

    expect(sub).not.toBeNull()
    expect(sub?.closest('[data-slot="dialog-content"]')).toBeNull()
    expect(sub?.ownerDocument.body.contains(sub)).toBe(true)
    expect(sub?.className).toContain('z-50')
    expect(sub?.className).not.toContain('z-(--z-modal-popover)')
  })

  it('portals a submenu into the dialog content and raises it above the parent menu', () => {
    render(
      <Dialog open>
        <DialogContent>
          <OpenSubmenu />
        </DialogContent>
      </Dialog>
    )

    const sub = screen.getByText('High').closest('[data-slot="dropdown-menu-sub-content"]')
    const dialog = screen.getByRole('dialog')
    const menu = screen.getByRole('menuitem', { name: 'Model' }).closest('[data-slot="dropdown-menu-content"]')

    expect(dialog.contains(sub)).toBe(true)
    expect(dialog.contains(menu)).toBe(true)
    expect(sub?.className).toContain('z-(--z-modal-popover)')
    expect((sub as HTMLElement).style.pointerEvents).toBe('auto')
  })

  it('uses the same explicit portal container as the parent menu', () => {
    function Hosted() {
      const [host, setHost] = useState<HTMLDivElement | null>(null)

      return (
        <>
          <div ref={setHost} />
          {host ? <OpenSubmenu portalContainer={host} /> : null}
        </>
      )
    }

    const { container } = render(<Hosted />)

    const sub = screen.getByText('High').closest('[data-slot="dropdown-menu-sub-content"]')
    const menu = screen.getByRole('menuitem', { name: 'Model' }).closest('[data-slot="dropdown-menu-content"]')
    const host = container.firstElementChild

    expect(host?.contains(sub)).toBe(true)
    expect(host?.contains(menu)).toBe(true)
    expect(sub?.className).toContain('z-(--z-modal-popover)')
  })

  it('selects a submenu row inside a dialog without dismissing the parent menu', async () => {
    const onMenuOpenChange = vi.fn()
    const onSelect = vi.fn((event: Event) => event.preventDefault())

    function EffortMenu({ subOpen }: { subOpen: boolean }) {
      return (
        <Dialog open>
          <DialogContent>
            <DropdownMenu onOpenChange={onMenuOpenChange} open>
              <DropdownMenuContent>
                <DropdownMenuSub open={subOpen}>
                  <DropdownMenuSubTrigger>Model</DropdownMenuSubTrigger>
                  <DropdownMenuSubContent>
                    <DropdownMenuItem onSelect={onSelect}>High</DropdownMenuItem>
                  </DropdownMenuSubContent>
                </DropdownMenuSub>
              </DropdownMenuContent>
            </DropdownMenu>
          </DialogContent>
        </Dialog>
      )
    }

    // Open the submenu after the parent menu, as a hover would.
    const { rerender } = render(<EffortMenu subOpen={false} />)
    rerender(<EffortMenu subOpen />)

    // DismissableLayer registers its document pointerdown listener in a
    // setTimeout(0); flush it so an outside press would actually dismiss.
    await act(() => new Promise(resolve => setTimeout(resolve, 10)))

    const row = screen.getByRole('menuitem', { name: 'High' })
    fireEvent.pointerDown(row, { button: 0, pointerType: 'mouse' })
    fireEvent.pointerUp(row, { button: 0, pointerType: 'mouse' })
    fireEvent.click(row)

    expect(onSelect).toHaveBeenCalledTimes(1)
    expect(onMenuOpenChange).not.toHaveBeenCalledWith(false)
  })
})
