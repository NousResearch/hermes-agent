import { cleanup, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { useWheelScrollHoverRest, WHEEL_SCROLL_HOVER_REST_MS } from './use-wheel-scroll-hover-rest'

let viewport: HTMLDivElement
let content: HTMLDivElement

const resting = () => content.hasAttribute('data-wheel-scrolling')
const wheel = (init: WheelEventInit = {}) => viewport.dispatchEvent(new WheelEvent('wheel', init))
const scroll = () => viewport.dispatchEvent(new Event('scroll'))

const pointer = (type: string, x: number, y: number) =>
  viewport.dispatchEvent(new MouseEvent(type, { clientX: x, clientY: y }))

function mount(paneVisible = true) {
  return renderHook(() =>
    useWheelScrollHoverRest({ contentRef: { current: content }, paneVisible, scrollRef: { current: viewport } })
  )
}

beforeEach(() => {
  vi.useFakeTimers()
  viewport = window.document.createElement('div')
  content = window.document.createElement('div')
  viewport.append(content)
})

afterEach(() => {
  cleanup()
  vi.useRealTimers()
})

it('rests the content while the wheel scrolls the transcript, and wakes it when the wheel stops', () => {
  mount()

  wheel()
  scroll()
  expect(resting()).toBe(true)

  // Still wheeling: each scroll pushes the wake back.
  vi.advanceTimersByTime(WHEEL_SCROLL_HOVER_REST_MS - 10)
  wheel()
  scroll()
  vi.advanceTimersByTime(WHEEL_SCROLL_HOVER_REST_MS - 10)
  expect(resting()).toBe(true)

  vi.advanceTimersByTime(20)
  expect(resting()).toBe(false)
})

it('leaves the content alone when the transcript scrolls with no wheel (a reply streaming in)', () => {
  mount()

  scroll()
  expect(resting()).toBe(false)

  // A wheel long before this scroll is not what moved it.
  wheel()
  vi.advanceTimersByTime(1000)
  scroll()
  expect(resting()).toBe(false)
})

it('leaves the content alone when the wheel scrolled something inside it, not the transcript', () => {
  mount()

  // A code block took the wheel: the viewport never fires a scroll.
  wheel()
  vi.advanceTimersByTime(WHEEL_SCROLL_HOVER_REST_MS)
  expect(resting()).toBe(false)
})

it('wakes at once when the pointer really moves or presses, but not when its position is re-sent', () => {
  mount()
  pointer('pointermove', 10, 10)

  wheel()
  scroll()
  pointer('pointermove', 10, 10)
  expect(resting()).toBe(true)

  pointer('pointermove', 12, 10)
  expect(resting()).toBe(false)

  // The wheel event that follows a real move belongs to a gesture the user has left.
  scroll()
  expect(resting()).toBe(false)

  wheel()
  scroll()
  expect(resting()).toBe(true)
  pointer('pointerdown', 12, 10)
  expect(resting()).toBe(false)
})

it('does not rest the content under a drag, and lets go of it when the pane hides', () => {
  const { unmount } = mount()

  wheel({ buttons: 1 })
  scroll()
  expect(resting()).toBe(false)

  wheel()
  scroll()
  expect(resting()).toBe(true)
  unmount()
  expect(resting()).toBe(false)
})
