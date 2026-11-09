import { afterEach, describe, expect, it, vi } from 'vitest'

import { forwardRailWheel, syncRailGutter, timelineRailStyle } from './timeline-layout'

/** Regression for #135098: the rail must not swallow the transcript's scrollbar or wheel. */

afterEach(() => {
  document.body.innerHTML = ''
  vi.restoreAllMocks()
})

const box = (element: HTMLElement, { client = 300, scroll = 300, top = 0 } = {}) => {
  Object.defineProperty(element, 'clientHeight', { configurable: true, value: client })
  Object.defineProperty(element, 'scrollHeight', { configurable: true, value: scroll })
  element.scrollTop = top

  return element
}

const setup = (ticks = { client: 280, scroll: 280, top: 0 }) => {
  const viewport = box(document.createElement('div'), { client: 600, scroll: 6000, top: 1000 })
  const rail = box(document.createElement('div'), ticks)
  const scrollBy = vi.fn()
  viewport.scrollBy = scrollBy as typeof viewport.scrollBy
  document.body.append(viewport, rail)

  return { rail, scrollBy, viewport }
}

const wheel = (init: WheelEventInit) => new WheelEvent('wheel', { cancelable: true, ...init })

describe('timelineRailStyle', () => {
  it('stands the rail left of the viewport scrollbar', () => {
    expect(timelineRailStyle(4)).toMatchObject({ position: 'absolute', right: 'var(--timeline-gutter, 0px)' })
  })
})

describe('syncRailGutter', () => {
  it('follows the viewport scrollbar as it comes and goes', () => {
    const rail = document.createElement('div')
    const viewport = document.createElement('div')
    const offsetWidth = vi.spyOn(viewport, 'offsetWidth', 'get').mockReturnValue(808)
    const clientWidth = vi.spyOn(viewport, 'clientWidth', 'get').mockReturnValue(800)

    syncRailGutter(rail, viewport)
    expect(rail.style.getPropertyValue('--timeline-gutter')).toBe('8px')

    offsetWidth.mockReturnValue(800)
    clientWidth.mockReturnValue(800)
    syncRailGutter(rail, viewport)
    expect(rail.style.getPropertyValue('--timeline-gutter')).toBe('0px')
  })
})

describe('forwardRailWheel', () => {
  it('scrolls the transcript when the rail has nothing to scroll', () => {
    const { rail, scrollBy, viewport } = setup()
    const seen = vi.fn()
    viewport.addEventListener('wheel', event => seen(event.deltaY))

    expect(forwardRailWheel(wheel({ deltaY: 120 }), rail, viewport)).toBe(true)
    expect(seen).toHaveBeenCalledWith(120)
    expect(scrollBy).toHaveBeenCalledWith({ top: 120 })
  })

  it('converts line and page deltas to pixels', () => {
    const { rail, scrollBy, viewport } = setup()

    forwardRailWheel(wheel({ deltaMode: WheelEvent.DOM_DELTA_LINE, deltaY: -3 }), rail, viewport)
    forwardRailWheel(wheel({ deltaMode: WheelEvent.DOM_DELTA_PAGE, deltaY: 1 }), rail, viewport)

    expect(scrollBy).toHaveBeenNthCalledWith(1, { top: -48 })
    expect(scrollBy).toHaveBeenNthCalledWith(2, { top: 600 })
  })

  it('lets a long rail scroll its own ticks, then hands over at either end', () => {
    const { rail, scrollBy, viewport } = setup({ client: 280, scroll: 1400, top: 500 })

    expect(forwardRailWheel(wheel({ deltaY: 100 }), rail, viewport)).toBe(false)
    expect(forwardRailWheel(wheel({ deltaY: -100 }), rail, viewport)).toBe(false)

    rail.scrollTop = 1120
    expect(forwardRailWheel(wheel({ deltaY: 100 }), rail, viewport)).toBe(true)

    rail.scrollTop = 0
    expect(forwardRailWheel(wheel({ deltaY: -100 }), rail, viewport)).toBe(true)
    expect(scrollBy).toHaveBeenCalledTimes(2)
  })

  it('leaves zoom gestures, horizontal wheels and cancelled wheels alone', () => {
    const { rail, scrollBy, viewport } = setup()

    expect(forwardRailWheel(wheel({ ctrlKey: true, deltaY: 100 }), rail, viewport)).toBe(false)
    expect(forwardRailWheel(wheel({ deltaX: 100 }), rail, viewport)).toBe(false)

    viewport.addEventListener('wheel', event => event.preventDefault())
    expect(forwardRailWheel(wheel({ deltaY: 100 }), rail, viewport)).toBe(false)
    expect(scrollBy).not.toHaveBeenCalled()
  })

  it('does not bubble the copy to ancestors', () => {
    const { rail, viewport } = setup()
    const outer = vi.fn()
    document.body.addEventListener('wheel', outer)

    forwardRailWheel(wheel({ deltaY: 100 }), rail, viewport)

    expect(outer).not.toHaveBeenCalled()
  })
})
