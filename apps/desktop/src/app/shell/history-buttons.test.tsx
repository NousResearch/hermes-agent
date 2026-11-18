import { cleanup, fireEvent, render, renderHook } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { keybindAction } from '@/lib/keybinds/actions'

import { $workspaceIsPage, NEW_CHAT_ROUTE } from '../routes'

import { $historyPosition, useMouseHistoryButtons } from './history-position'

vi.mock('@/lib/host-capabilities', () => ({ hostHas: () => true }))

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

describe('mouse back/forward buttons', () => {
  it('walk the window history in the redesign', () => {
    const back = vi.spyOn(window.history, 'back').mockImplementation(() => undefined)
    const forward = vi.spyOn(window.history, 'forward').mockImplementation(() => undefined)
    renderHook(() => useMouseHistoryButtons(true, vi.fn()))

    fireEvent.mouseDown(window.document.body, { button: 3 })
    fireEvent.mouseDown(window.document.body, { button: 4 })
    fireEvent.mouseDown(window.document.body, { button: 0 })

    expect(back).toHaveBeenCalledOnce()
    expect(forward).toHaveBeenCalledOnce()
  })

  it('leave a press the preview pane already claimed for its own page', () => {
    const back = vi.spyOn(window.history, 'back').mockImplementation(() => undefined)
    renderHook(() => useMouseHistoryButtons(true, vi.fn()))
    const { getByTestId } = render(<aside data-testid="preview" onMouseDown={event => event.preventDefault()} />)

    fireEvent.mouseDown(getByTestId('preview'), { button: 3 })

    expect(back).not.toHaveBeenCalled()
  })

  it('take the band Back’s way out of a workspace page with nothing behind it', () => {
    const back = vi.spyOn(window.history, 'back').mockImplementation(() => undefined)
    const navigate = vi.fn()
    $historyPosition.set({ index: 0, last: 0 })
    $workspaceIsPage.set(true)
    renderHook(() => useMouseHistoryButtons(true, navigate))

    fireEvent.mouseDown(window.document.body, { button: 3 })

    expect(navigate).toHaveBeenCalledWith(NEW_CHAT_ROUTE)
    expect(back).not.toHaveBeenCalled()
    $workspaceIsPage.set(false)
  })

  it('do nothing in the classic shell', () => {
    const back = vi.spyOn(window.history, 'back').mockImplementation(() => undefined)
    renderHook(() => useMouseHistoryButtons(false, vi.fn()))

    fireEvent.mouseDown(window.document.body, { button: 3 })

    expect(back).not.toHaveBeenCalled()
  })
})

describe('Back/Forward shortcuts', () => {
  it('are rows of the shortcut map, in Navigation', () => {
    expect(keybindAction('nav.back')?.category).toBe('navigation')
    expect(keybindAction('nav.forward')?.category).toBe('navigation')
  })
})
