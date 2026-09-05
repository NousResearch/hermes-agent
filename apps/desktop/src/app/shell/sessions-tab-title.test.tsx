import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { SessionsTabTitle } from './sessions-tab-title'

// The Sessions pane tab carries the unread count as a visible node plus an
// accessible label. Zero must render no count node at all (not "0" — the tab
// is just "sessions"); any nonzero count renders the number with both
// an aria-label so assistive tech names the count action.

const renderTitle = (unread: number, onOpenNextUnread = vi.fn()) =>
  render(<SessionsTabTitle onOpenNextUnread={onOpenNextUnread} unread={unread} />)

afterEach(() => {
  cleanup()
})

describe('SessionsTabTitle', () => {
  it('renders only the sessions label when nothing is unread', () => {
    const { container } = renderTitle(0)

    expect(screen.getByText('sessions')).toBeDefined()
    expect(container.querySelector('[aria-label]')).toBeNull()
    expect(container.textContent).not.toContain('0')
  })

  it('exposes the visible count with an accessible label when sessions are unread', () => {
    renderTitle(12)

    const count = screen.getByRole('button', { name: '12 unread sessions' })
    expect(count.textContent).toBe('12')
    expect(screen.getByText('sessions')).toBeDefined()
  })

  it('labels a single unread session in singular', () => {
    renderTitle(1)

    expect(screen.getByLabelText('1 unread session').textContent).toBe('1')
  })

  it('reserves two digit slots so crossing 9 → 10 does not reflow the tab', () => {
    // The count span keeps its box through the 1-to-2 digit transition; the
    // reserved width is asserted on the class contract rather than layout.
    const single = renderTitle(9)
    const singleCount = screen.getByLabelText('9 unread sessions') as HTMLElement
    expect(singleCount.className).toContain('min-w-[2ch]')
    single.unmount()
    cleanup()

    renderTitle(10)
    const doubleCount = screen.getByLabelText('10 unread sessions') as HTMLElement
    expect(doubleCount.className).toBe(singleCount.className)
  })

  it('opens the newest unread without turning the count into a pane drag or tab select', () => {
    const onOpenNextUnread = vi.fn()
    const onParentClick = vi.fn()
    const onParentPointerDown = vi.fn()

    render(
      <div onClick={onParentClick} onPointerDown={onParentPointerDown}>
        <SessionsTabTitle onOpenNextUnread={onOpenNextUnread} unread={3} />
      </div>
    )

    const count = screen.getByRole('button', { name: '3 unread sessions' })
    fireEvent.pointerDown(count)
    fireEvent.click(count)

    expect(onOpenNextUnread).toHaveBeenCalledOnce()
    expect(onParentPointerDown).not.toHaveBeenCalled()
    expect(onParentClick).not.toHaveBeenCalled()
  })
})
