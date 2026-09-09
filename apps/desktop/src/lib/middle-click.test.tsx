import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { middleClickHandlers } from './middle-click'

afterEach(cleanup)

/** A middle click as a real three-button mouse delivers it. Chromium on
 *  Windows/Linux swallows the trailing `auxclick` when the press starts
 *  autoscroll, so the gesture may NOT depend on that event. */
function middleClick(element: Element, upOn: Element = element) {
  fireEvent.mouseDown(element, { button: 1 })
  fireEvent.pointerDown(element, { button: 1 })
  fireEvent.pointerUp(upOn, { button: 1 })
}

function Target({ action, id = 'target' }: { action?: () => void; id?: string }) {
  return (
    <button {...middleClickHandlers(action)} id={id} type="button">
      {id}
    </button>
  )
}

describe('middleClickHandlers', () => {
  it('fires without an auxclick — the event Chromium eats when autoscroll starts', () => {
    const action = vi.fn()
    render(<Target action={action} />)

    middleClick(screen.getByText('target'))
    expect(action).toHaveBeenCalledTimes(1)
  })

  it('cancels mousedown so the autoscroll pan widget never appears', () => {
    render(<Target action={vi.fn()} />)

    const down = fireEvent.mouseDown(screen.getByText('target'), { button: 1 })
    expect(down).toBe(false) // preventDefault() called
  })

  it('cancels the middle mousedown even with no action — the surface owns the button', () => {
    render(<Target />)

    expect(fireEvent.mouseDown(screen.getByText('target'), { button: 1 })).toBe(false)
  })

  it('ignores left and right buttons', () => {
    const action = vi.fn()
    render(<Target action={action} />)

    const target = screen.getByText('target')
    fireEvent.pointerDown(target, { button: 0 })
    fireEvent.pointerUp(target, { button: 0 })
    fireEvent.pointerDown(target, { button: 2 })
    fireEvent.pointerUp(target, { button: 2 })
    expect(action).not.toHaveBeenCalled()
  })

  it('does nothing when the release lands on a different element', () => {
    const pressed = vi.fn()
    const released = vi.fn()
    render(
      <>
        <Target action={pressed} id="pressed" />
        <Target action={released} id="released" />
      </>
    )

    middleClick(screen.getByText('pressed'), screen.getByText('released'))
    expect(pressed).not.toHaveBeenCalled()
    expect(released).not.toHaveBeenCalled()
  })

  it('a press with no action cannot arm the NEXT element it releases over', () => {
    const action = vi.fn()
    render(
      <>
        <Target id="inert" />
        <Target action={action} id="live" />
      </>
    )

    middleClick(screen.getByText('inert'), screen.getByText('live'))
    expect(action).not.toHaveBeenCalled()
  })
})
