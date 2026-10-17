import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { createRef } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { PaneTab, PaneTabLabel, PaneTabStrip } from './pane-tab'

afterEach(cleanup)

describe('PaneTabStrip wheel scrolling', () => {
  // jsdom has no layout, so give the tablist an overflowing geometry by hand.
  const renderStrip = ({ clientWidth = 300, scrollWidth = 900 } = {}) => {
    const listRef = createRef<HTMLDivElement>()
    render(
      <PaneTabStrip listRef={listRef}>
        <PaneTab>
          <PaneTabLabel>tab</PaneTabLabel>
        </PaneTab>
      </PaneTabStrip>
    )

    const list = screen.getByRole('tablist')
    Object.defineProperty(list, 'clientWidth', { configurable: true, value: clientWidth })
    Object.defineProperty(list, 'scrollWidth', { configurable: true, value: scrollWidth })

    return { list, listRef }
  }

  const wheel = (el: HTMLElement, init: WheelEventInit) => {
    const event = new WheelEvent('wheel', { bubbles: true, cancelable: true, ...init })
    el.dispatchEvent(event)

    return event
  }

  it('a plain vertical wheel scrolls an overflowing strip sideways — no Shift needed', () => {
    const { list, listRef } = renderStrip()
    // The scroller still reaches the caller's ref (useActiveTabVisible reads it).
    expect(listRef.current).toBe(list)

    const down = wheel(list, { deltaY: 120 })
    expect(list.scrollLeft).toBe(120)
    expect(down.defaultPrevented).toBe(true)

    wheel(list, { deltaY: -40 })
    expect(list.scrollLeft).toBe(80)
  })

  it('leaves horizontal wheels, Ctrl/⌘ zoom and a fitting strip to the browser', () => {
    const { list } = renderStrip()

    for (const init of [
      { deltaX: 120, deltaY: 10 },
      { ctrlKey: true, deltaY: 120 },
      { deltaY: 120, metaKey: true }
    ]) {
      expect(wheel(list, init).defaultPrevented).toBe(false)
    }

    expect(list.scrollLeft).toBe(0)

    Object.defineProperty(list, 'clientWidth', { configurable: true, value: 900 })
    expect(wheel(list, { deltaY: 120 }).defaultPrevented).toBe(false)
    expect(list.scrollLeft).toBe(0)
  })
})

describe('PaneTab close gestures', () => {
  it('middle-click closes — pointer events only, no auxclick', () => {
    const onClose = vi.fn()
    render(
      <PaneTab onClose={onClose}>
        <PaneTabLabel>tab</PaneTabLabel>
      </PaneTab>
    )

    const tab = screen.getByText('tab')
    fireEvent.pointerDown(tab, { button: 1 })
    fireEvent.pointerUp(tab, { button: 1 })
    expect(onClose).toHaveBeenCalledTimes(1)
  })

  it('⌘-click (metaKey + button 0) closes — the Mac middle-click equivalent', () => {
    const onClose = vi.fn()
    render(
      <PaneTab onClose={onClose}>
        <PaneTabLabel>tab</PaneTabLabel>
      </PaneTab>
    )

    fireEvent.pointerDown(screen.getByText('tab'), { button: 0, metaKey: true })
    expect(onClose).toHaveBeenCalledTimes(1)
  })

  it('⌘-click preempts the shell drag/activate pointerdown handler', () => {
    const onClose = vi.fn()
    const onPointerDown = vi.fn()
    render(
      <PaneTab onClose={onClose} onPointerDown={onPointerDown}>
        <PaneTabLabel>tab</PaneTabLabel>
      </PaneTab>
    )

    fireEvent.pointerDown(screen.getByText('tab'), { button: 0, metaKey: true })
    expect(onClose).toHaveBeenCalledTimes(1)
    expect(onPointerDown).not.toHaveBeenCalled()
  })

  it('⌘-click swallows the follow-up activation click (capture phase)', () => {
    const onClose = vi.fn()
    const onActivate = vi.fn()
    render(
      <PaneTab onClose={onClose}>
        <PaneTabLabel as="button" onClick={onActivate}>
          tab
        </PaneTabLabel>
      </PaneTab>
    )

    fireEvent.click(screen.getByText('tab'), { button: 0, metaKey: true })
    expect(onActivate).not.toHaveBeenCalled()
  })

  it('plain left-click neither closes nor blocks activation', () => {
    const onClose = vi.fn()
    const onActivate = vi.fn()
    const onPointerDown = vi.fn()
    render(
      <PaneTab onClose={onClose} onPointerDown={onPointerDown}>
        <PaneTabLabel as="button" onClick={onActivate}>
          tab
        </PaneTabLabel>
      </PaneTab>
    )

    fireEvent.pointerDown(screen.getByText('tab'), { button: 0 })
    fireEvent.click(screen.getByText('tab'), { button: 0 })
    expect(onClose).not.toHaveBeenCalled()
    expect(onPointerDown).toHaveBeenCalledTimes(1)
    expect(onActivate).toHaveBeenCalledTimes(1)
  })

  it('does nothing without an onClose (uncloseable workspace tab)', () => {
    const onPointerDown = vi.fn()
    render(
      <PaneTab onPointerDown={onPointerDown}>
        <PaneTabLabel>tab</PaneTabLabel>
      </PaneTab>
    )

    fireEvent.pointerDown(screen.getByText('tab'), { button: 0, metaKey: true })
    expect(onPointerDown).toHaveBeenCalledTimes(1)
  })
})

describe('PaneTab hover close button', () => {
  it('clicking the ✕ closes without activating or dragging the tab', () => {
    const onClose = vi.fn()
    const onActivate = vi.fn()
    const onPointerDown = vi.fn()
    render(
      <PaneTab onClose={onClose} onPointerDown={onPointerDown}>
        <PaneTabLabel as="button" onClick={onActivate}>
          tab
        </PaneTabLabel>
      </PaneTab>
    )

    const close = screen.getByRole('button', { name: 'Close' })
    fireEvent.pointerDown(close, { button: 0 })
    fireEvent.click(close, { button: 0 })
    expect(onClose).toHaveBeenCalledTimes(1)
    expect(onActivate).not.toHaveBeenCalled()
    expect(onPointerDown).not.toHaveBeenCalled()
  })

  it('renders no ✕ without an onClose', () => {
    render(
      <PaneTab>
        <PaneTabLabel>tab</PaneTabLabel>
      </PaneTab>
    )

    expect(screen.queryByRole('button', { name: 'Close' })).toBeNull()
  })

  it('renders no ✕ on a vertical rail tab (middle/⌘-click only there)', () => {
    const onClose = vi.fn()
    render(
      <PaneTab onClose={onClose} vertical>
        <PaneTabLabel>tab</PaneTabLabel>
      </PaneTab>
    )

    expect(screen.queryByRole('button', { name: 'Close' })).toBeNull()
  })
})
