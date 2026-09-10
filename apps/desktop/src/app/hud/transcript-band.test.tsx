import { act, render } from '@testing-library/react'
import { useRef } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { stubResizeObserver } from '@/test/jsdom'

import { useHudTranscriptBand } from './transcript-band'

function Harness({ withViewport }: { withViewport: boolean }) {
  const ref = useRef<HTMLDivElement | null>(null)

  useHudTranscriptBand(ref)

  return (
    <div ref={ref}>
      <div data-slot="composer-dock" />
      {withViewport && (
        <div data-slot="aui_thread-viewport">
          <div data-slot="aui_thread-content">
            <div>row</div>
          </div>
        </div>
      )}
    </div>
  )
}

beforeEach(() => {
  stubResizeObserver()
  vi.useFakeTimers()
})

afterEach(() => {
  vi.useRealTimers()
})

describe('useHudTranscriptBand', () => {
  // Regression for #107050: after every TurnRow got `data-slot="aui_message-group"`
  // the selector stopped matching anything and the band collapsed to 0px. A
  // fixture with slot-less rows (what the old code expected) wouldn't see it,
  // so we verify the app's current DOM shape produces a non-zero band.
  it('sizes the band when message rows carry aui_message-group slot', () => {
    const measureSpy = vi.spyOn(HTMLElement.prototype, 'getBoundingClientRect')
    const stubY = 100
    const stubHeight = 710

    measureSpy.mockImplementation(function (this: HTMLElement) {
      const el = this
      // The row: bottom 710px beyond its top (a realistic turn row)
      if (el.getAttribute('data-slot') === 'aui_message-group') {
        return { top: stubY, bottom: stubY + stubHeight, height: stubHeight, left: 0, right: 0, width: 0, x: 0, y: stubY, toJSON: () => ({}) } as DOMRect
      }
      // Composer dock: arbitrary
      if (el.getAttribute('data-slot') === 'composer-dock') {
        return { top: 0, bottom: 64, height: 64, left: 0, right: 0, width: 0, x: 0, y: 0, toJSON: () => ({}) } as DOMRect
      }
      // Everything else: zero
      return { top: 0, bottom: 0, height: 0, left: 0, right: 0, width: 0, x: 0, y: 0, toJSON: () => ({}) } as DOMRect
    })

    function TestComponent() {
      const ref = useRef<HTMLDivElement | null>(null)
      useHudTranscriptBand(ref)

      return (
        <div ref={ref}>
          <div data-slot="composer-dock" />
          <div data-slot="aui_thread-viewport">
            <div data-slot="aui_thread-content">
              <div data-slot="aui_message-group">row</div>
            </div>
          </div>
        </div>
      )
    }

    const { container } = render(<TestComponent />)

    act(() => vi.advanceTimersByTime(100))

    // Not testing the exact pixel value (that's what layout.test.ts owns), but
    // the hook must recognize the row and compute a non-zero band. Before #107050
    // this was 0px: rows = [], contentSpan = 0, hudTranscriptHeight(...) -> 0.
    const rootDiv = container.firstElementChild as HTMLElement
    expect(rootDiv?.style.getPropertyValue('--hud-band-height')).not.toBe('0px')
    expect(rootDiv?.style.getPropertyValue('--hud-band-height')).not.toBe('')
  })

  // The bug this replaced: the probe polled every 500ms for the lifetime of
  // the HUD window, duplicating every measurement the ResizeObserver already
  // owned once the viewport existed — a permanent idle timer firing re-renders
  // forever instead of the "poll briefly, then hand off" the code documented.
  it('stops polling once the viewport mounts', () => {
    const measureSpy = vi.spyOn(HTMLElement.prototype, 'getBoundingClientRect')
    const { rerender } = render(<Harness withViewport={false} />)
    const beforeWaiting = measureSpy.mock.calls.length

    act(() => vi.advanceTimersByTime(500))
    act(() => vi.advanceTimersByTime(500))
    const whileWaiting = measureSpy.mock.calls.length

    expect(whileWaiting).toBeGreaterThan(beforeWaiting)

    rerender(<Harness withViewport />)
    act(() => vi.advanceTimersByTime(500))
    const justAfterFound = measureSpy.mock.calls.length

    expect(justAfterFound).toBeGreaterThan(whileWaiting)

    act(() => vi.advanceTimersByTime(10_000))
    const muchLater = measureSpy.mock.calls.length

    expect(muchLater).toBe(justAfterFound)
  })
})
