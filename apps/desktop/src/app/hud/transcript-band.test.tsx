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
