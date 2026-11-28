import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import type { RefObject } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { PaneTab, PaneTabLabel, PaneTabStrip } from './pane-tab'
import { tabStripEdges, tabStripWheelDelta } from './pane-tab-scroll'

afterEach(cleanup)

/** jsdom has no layout: hand the scroller the numbers a real strip would report. */
function layout(el: HTMLElement, { clientWidth = 300, scrollLeft = 0, scrollWidth = 900 } = {}) {
  let left = scrollLeft

  Object.defineProperty(el, 'clientWidth', { configurable: true, value: clientWidth })
  Object.defineProperty(el, 'scrollWidth', { configurable: true, value: scrollWidth })
  Object.defineProperty(el, 'scrollLeft', {
    configurable: true,
    get: () => left,
    set: (next: number) => {
      left = next
    }
  })
  el.scrollBy = vi.fn()

  return el
}

const renderStrip = (listRef?: RefObject<HTMLDivElement | null>) =>
  render(
    <PaneTabStrip listRef={listRef}>
      <PaneTab role="tab">
        <PaneTabLabel>one</PaneTabLabel>
      </PaneTab>
      <PaneTab role="tab">
        <PaneTabLabel>two</PaneTabLabel>
      </PaneTab>
      <PaneTab role="tab">
        <PaneTabLabel>three</PaneTabLabel>
      </PaneTab>
    </PaneTabStrip>
  )

const scrollerOf = () => screen.getByRole('tablist')

/** Re-measure the way the strip learns about a new scroll position. */
const settle = (el: HTMLElement) => fireEvent.scroll(el)

describe('tabStripEdges', () => {
  const edges = (scrollLeft: number) => tabStripEdges({ clientWidth: 300, rtl: false, scrollLeft, scrollWidth: 900 })

  it('reports no cue when every tab fits', () => {
    expect(tabStripEdges({ clientWidth: 900, rtl: false, scrollLeft: 0, scrollWidth: 900 })).toEqual({
      left: false,
      right: false
    })
  })

  it('shows only the trailing cue at the start of the strip', () => {
    expect(edges(0)).toEqual({ left: false, right: true })
  })

  it('shows both cues in the middle', () => {
    expect(edges(200)).toEqual({ left: true, right: true })
  })

  it('shows only the leading cue at the end', () => {
    expect(edges(600)).toEqual({ left: true, right: false })
  })

  it('reads RTL offsets from the logical start (0) down to -max', () => {
    const rtl = (scrollLeft: number) => tabStripEdges({ clientWidth: 300, rtl: true, scrollLeft, scrollWidth: 900 })

    // Parked on the first tab — the logical start, physically on the right.
    expect(rtl(0)).toEqual({ left: true, right: false })
    expect(rtl(-600)).toEqual({ left: false, right: true })
  })
})

describe('tabStripWheelDelta', () => {
  const wheel = (init: WheelEventInit) => new WheelEvent('wheel', init)

  it('maps a plain vertical wheel onto the strip', () => {
    expect(tabStripWheelDelta(wheel({ deltaY: 120 }), 300)).toBe(120)
    expect(tabStripWheelDelta(wheel({ deltaY: -120 }), 300)).toBe(-120)
  })

  it('leaves a horizontal gesture to the browser', () => {
    expect(tabStripWheelDelta(wheel({ deltaX: 120, deltaY: 10 }), 300)).toBe(0)
  })

  it('never eats zoom', () => {
    expect(tabStripWheelDelta(wheel({ ctrlKey: true, deltaY: 120 }), 300)).toBe(0)
    expect(tabStripWheelDelta(wheel({ deltaY: 120, metaKey: true }), 300)).toBe(0)
  })

  it('scales line and page deltas', () => {
    expect(tabStripWheelDelta(wheel({ deltaMode: 1, deltaY: 3 }), 300)).toBe(48)
    expect(tabStripWheelDelta(wheel({ deltaMode: 2, deltaY: 1 }), 300)).toBe(300)
  })
})

describe('PaneTabStrip overflow cues', () => {
  it('shows the trailing cue only while tabs are clipped and scrolls a page per press', () => {
    const listRef = { current: null as HTMLDivElement | null }

    renderStrip(listRef)

    const scroller = scrollerOf() as HTMLDivElement
    layout(scroller)

    // The caller's ref still receives the scroller (`useActiveTabVisible` writes through it).
    expect(listRef.current).toBe(scroller)

    settle(scroller)
    expect(screen.queryByRole('button', { name: 'Scroll tabs left' })).toBeNull()

    fireEvent.click(screen.getByRole('button', { name: 'Scroll tabs right' }))
    expect(scroller.scrollBy).toHaveBeenCalledWith({ behavior: expect.any(String), left: 240 })

    // Once scrolled into the middle both ends have hidden tabs.
    scroller.scrollLeft = 200
    settle(scroller)
    expect(screen.getByRole('button', { name: 'Scroll tabs left' })).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: 'Scroll tabs left' }))
    expect(scroller.scrollBy).toHaveBeenLastCalledWith({ behavior: expect.any(String), left: -240 })
  })

  it('keeps a strip that fits clean — no cues', () => {
    renderStrip()

    const scroller = scrollerOf()
    layout(scroller, { clientWidth: 900, scrollWidth: 900 })
    settle(scroller)

    expect(screen.queryByRole('button', { name: 'Scroll tabs left' })).toBeNull()
    expect(screen.queryByRole('button', { name: 'Scroll tabs right' })).toBeNull()
  })

  it('scrolls sideways with a plain wheel and swallows only the gestures it handles', () => {
    renderStrip()

    const scroller = scrollerOf() as HTMLDivElement
    layout(scroller, { clientWidth: 300, scrollLeft: 0, scrollWidth: 900 })

    const down = new WheelEvent('wheel', { bubbles: true, cancelable: true, deltaY: 120 })
    scroller.dispatchEvent(down)

    expect(down.defaultPrevented).toBe(true)
    expect(scroller.scrollLeft).toBe(120)

    // A horizontal gesture is the browser's own axis.
    const sideways = new WheelEvent('wheel', { bubbles: true, cancelable: true, deltaX: 120, deltaY: 0 })
    scroller.dispatchEvent(sideways)
    expect(sideways.defaultPrevented).toBe(false)
    expect(scroller.scrollLeft).toBe(120)
  })

  it('lets a cue press through as a click, not a strip drag', () => {
    const onPointerDown = vi.fn()

    render(
      <PaneTabStrip onPointerDown={onPointerDown}>
        <PaneTab role="tab">
          <PaneTabLabel>one</PaneTabLabel>
        </PaneTab>
      </PaneTabStrip>
    )

    const scroller = scrollerOf()
    layout(scroller)
    settle(scroller)

    const cue = screen.getByRole('button', { name: 'Scroll tabs right' })
    fireEvent.pointerDown(cue)

    expect(onPointerDown).not.toHaveBeenCalled()
  })
})
