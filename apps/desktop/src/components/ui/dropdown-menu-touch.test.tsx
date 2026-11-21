import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from './dropdown-menu'

beforeEach(() => {
  globalThis.document.documentElement.dataset.hermesDesktopHost = 'browser'
  vi.stubGlobal('matchMedia', vi.fn().mockReturnValue({ matches: true }))
  Element.prototype.hasPointerCapture ??= () => false
  Element.prototype.releasePointerCapture ??= () => undefined
  Element.prototype.scrollIntoView ??= () => undefined
})

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
  delete globalThis.document.documentElement.dataset.hermesDesktopHost
})

function Menu({ onOpenChange = vi.fn() }: { onOpenChange?: (open: boolean) => void }) {
  return (
    <DropdownMenu onOpenChange={onOpenChange}>
      <DropdownMenuTrigger asChild>
        <button type="button">Status menu</button>
      </DropdownMenuTrigger>
      <DropdownMenuContent>
        <DropdownMenuItem>Menu action</DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

describe('browser-hosted dropdown touch activation', () => {
  it('does not open when an upward swipe starts on the trigger', () => {
    const onOpenChange = vi.fn()
    render(<Menu onOpenChange={onOpenChange} />)
    const trigger = screen.getByRole('button', { name: 'Status menu' })

    fireEvent.pointerDown(trigger, { button: 0, clientY: 800, pointerId: 1, pointerType: 'touch' })
    expect(onOpenChange).not.toHaveBeenCalled()
    fireEvent.pointerMove(trigger, { clientY: 740, pointerId: 1, pointerType: 'touch' })
    fireEvent.pointerCancel(trigger, { pointerId: 1, pointerType: 'touch' })

    expect(screen.queryByRole('menu')).toBeNull()
    expect(onOpenChange).not.toHaveBeenCalled()

    fireEvent.pointerDown(trigger, { button: 0, pointerId: 2, pointerType: 'touch' })
    fireEvent.pointerUp(trigger, { button: 0, pointerId: 2, pointerType: 'touch' })
    fireEvent.click(trigger, { pointerType: 'touch' })
    expect(onOpenChange).toHaveBeenCalledExactlyOnceWith(true)
    expect(screen.getByRole('menuitem', { name: 'Menu action' })).toBeTruthy()
  })

  it('does not treat a drag on a non-scrolling trigger as a tap', () => {
    const onOpenChange = vi.fn()
    render(<Menu onOpenChange={onOpenChange} />)
    const trigger = screen.getByRole('button', { name: 'Status menu' })

    fireEvent.pointerDown(trigger, { button: 0, clientX: 50, clientY: 800, pointerId: 1, pointerType: 'touch' })
    fireEvent.pointerMove(trigger, { clientX: 50, clientY: 740, pointerId: 1, pointerType: 'touch' })
    fireEvent.pointerUp(trigger, { button: 0, clientX: 50, clientY: 740, pointerId: 1, pointerType: 'touch' })
    fireEvent.click(trigger, { pointerType: 'touch' })

    expect(onOpenChange).not.toHaveBeenCalled()
  })

  it('ignores a disabled asChild button even if pointer events arrive', () => {
    const onOpenChange = vi.fn()
    render(
      <DropdownMenu onOpenChange={onOpenChange}>
        <DropdownMenuTrigger asChild>
          <button disabled type="button">
            Disabled status menu
          </button>
        </DropdownMenuTrigger>
        <DropdownMenuContent>
          <DropdownMenuItem>Menu action</DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
    )
    const trigger = screen.getByRole('button', { name: 'Disabled status menu' })
    fireEvent.pointerDown(trigger, { button: 0, pointerType: 'touch' })
    fireEvent.pointerUp(trigger, { button: 0, pointerType: 'touch' })
    expect(onOpenChange).not.toHaveBeenCalled()
  })

  it.each([
    { down: { button: 2, pointerId: 1 }, up: { pointerId: 1 } },
    { down: { button: 0, ctrlKey: true, pointerId: 1 }, up: { pointerId: 1 } },
    { down: { button: 0, pointerId: 1 }, up: { pointerId: 2 } }
  ])('ignores ineligible presses and unrelated pointer releases ($down, $up)', ({ down, up }) => {
    const onOpenChange = vi.fn()
    render(<Menu onOpenChange={onOpenChange} />)
    const trigger = screen.getByRole('button', { name: 'Status menu' })
    fireEvent.pointerDown(trigger, { pointerType: 'touch', ...down })
    fireEvent.pointerUp(trigger, { pointerType: 'touch', ...up })
    expect(onOpenChange).not.toHaveBeenCalled()
  })

  it('keeps controlled menus and keyboard activation working after a canceled touch', () => {
    const onOpenChange = vi.fn()

    const view = (open: boolean) => (
      <DropdownMenu onOpenChange={onOpenChange} open={open}>
        <DropdownMenuTrigger>Status menu</DropdownMenuTrigger>
        <DropdownMenuContent>
          <DropdownMenuItem>Menu action</DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
    )

    const { rerender } = render(view(false))
    const trigger = screen.getByRole('button', { name: 'Status menu' })

    fireEvent.pointerDown(trigger, { button: 0, pointerType: 'touch' })
    fireEvent.pointerUp(trigger, { button: 0, pointerType: 'touch' })
    fireEvent.click(trigger)
    expect(onOpenChange).toHaveBeenCalledExactlyOnceWith(true)
    expect(screen.queryByRole('menu')).toBeNull()
    rerender(view(true))
    expect(screen.getByRole('menuitem', { name: 'Menu action' })).toBeTruthy()
    rerender(view(false))
    onOpenChange.mockClear()

    fireEvent.pointerDown(trigger, { button: 0, pointerType: 'touch' })
    fireEvent.pointerCancel(trigger, { pointerType: 'touch' })
    fireEvent.keyDown(trigger, { key: 'Enter' })
    fireEvent.click(trigger, { detail: 0 })
    expect(onOpenChange).toHaveBeenCalledExactlyOnceWith(true)
  })

  it.each([
    { browser: true, pointerType: 'mouse' },
    { browser: false, pointerType: 'touch' }
  ])('preserves pointer-down activation for $pointerType (browser=$browser)', ({ browser, pointerType }) => {
    if (!browser) {
      delete globalThis.document.documentElement.dataset.hermesDesktopHost
    }

    const onOpenChange = vi.fn()
    render(<Menu onOpenChange={onOpenChange} />)
    const trigger = screen.getByRole('button', { name: 'Status menu' })

    fireEvent.pointerDown(trigger, { button: 0, pointerType })
    fireEvent.pointerUp(trigger, { button: 0, pointerType })
    fireEvent.click(trigger, { pointerType })

    expect(onOpenChange).toHaveBeenCalledExactlyOnceWith(true)
  })
})
