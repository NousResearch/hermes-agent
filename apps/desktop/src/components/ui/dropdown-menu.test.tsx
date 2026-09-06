import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuSearch } from './dropdown-menu'

afterEach(cleanup)

it.each(['ArrowDown', 'ArrowUp'])('leaves %s with the IME rather than handing search focus to menu items', key => {
  const onKeyDown = vi.fn()
  render(<DropdownMenu open><DropdownMenuContent>
    <DropdownMenuSearch aria-label="Filter" onKeyDown={onKeyDown} />
    <DropdownMenuItem>Choice</DropdownMenuItem>
  </DropdownMenuContent></DropdownMenu>)
  const input = screen.getByRole('textbox', { name: 'Filter' })
  input.focus()
  const accepted = fireEvent.keyDown(input, { key, isComposing: true })
  expect(window.document.activeElement).toBe(input)
  expect(accepted).toBe(true)
  expect(onKeyDown).not.toHaveBeenCalled()
})
