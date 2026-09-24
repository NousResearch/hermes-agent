import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest'

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
