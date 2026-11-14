import { cleanup, fireEvent, render, renderHook } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { keybindAction } from '@/lib/keybinds/actions'

import { useMouseHistoryButtons } from './history-position'

vi.mock('@/lib/host-capabilities', () => ({ hostHas: () => true }))

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

describe('mouse back/forward buttons', () => {
  it('walk the window history in the redesign', () => {
    const back = vi.spyOn(window.history, 'back').mockImplementation(() => undefined)
    const forward = vi.spyOn(window.history, 'forward').mockImplementation(() => undefined)
    renderHook(() => useMouseHistoryButtons(true))

    fireEvent.mouseDown(window.document.body, { button: 3 })
    fireEvent.mouseDown(window.document.body, { button: 4 })
    fireEvent.mouseDown(window.document.body, { button: 0 })

    expect(back).toHaveBeenCalledOnce()
    expect(forward).toHaveBeenCalledOnce()
  })

  it('leave a press the preview pane already claimed for its own page', () => {
    const back = vi.spyOn(window.history, 'back').mockImplementation(() => undefined)
    renderHook(() => useMouseHistoryButtons(true))
    const { getByTestId } = render(<aside data-testid="preview" onMouseDown={event => event.preventDefault()} />)

    fireEvent.mouseDown(getByTestId('preview'), { button: 3 })

    expect(back).not.toHaveBeenCalled()
  })

  it('do nothing in the classic shell', () => {
    const back = vi.spyOn(window.history, 'back').mockImplementation(() => undefined)
    renderHook(() => useMouseHistoryButtons(false))

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
