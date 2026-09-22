import { act, render } from '@testing-library/react'
import { type ReactNode, useRef } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { stubResizeObserver } from '@/test/jsdom'

import { useHudTranscriptBand } from './transcript-band'

function Harness({ children, withViewport }: { children?: ReactNode; withViewport: boolean }) {
  const ref = useRef<HTMLDivElement | null>(null)

  useHudTranscriptBand(ref)

  return (
    <div ref={ref}>
      <div data-slot="composer-dock" />
      {withViewport && (
        <div data-slot="aui_thread-viewport">
          <div data-slot="aui_thread-content">{children ?? <div>row</div>}</div>
        </div>
      )}
    </div>
  )
}

function bandHeight(container: HTMLElement): number {
  const root = container.firstElementChild as HTMLElement

  return Number.parseFloat(root.style.getPropertyValue('--hud-band-height') || '0')
}

function stubMeasuredRects() {
  vi.spyOn(HTMLElement.prototype, 'getBoundingClientRect').mockImplementation(function (this: HTMLElement) {
    if (this.dataset.slot === 'composer-dock') {
      return { bottom: 768, height: 58, top: 710 } as DOMRect
    }

    return { bottom: 172, height: 72, top: 100 } as DOMRect
  })
}

beforeEach(() => {
  stubResizeObserver()
  vi.useFakeTimers()
})

afterEach(() => {
  vi.useRealTimers()
  vi.restoreAllMocks()
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

  // TurnRow roots now carry data-slot="aui_message-group". The band must still
  // treat them as transcript rows; matching only *:not([data-slot]) leaves a
  // short session at --hud-band-height: 0px and clips the transcript.
  it('measures a TurnRow with data-slot="aui_message-group" as transcript height', () => {
    stubMeasuredRects()

    const { container } = render(
      <Harness withViewport>
        <div data-slot="aui_message-group">turn</div>
      </Harness>
    )

    expect(bandHeight(container)).toBeGreaterThan(0)
  })

  it('does not treat clearance or background-resume spacers as message rows', () => {
    stubMeasuredRects()

    const { container } = render(
      <Harness withViewport>
        <div data-slot="aui_background-resume">resume</div>
        <div data-slot="aui_composer-clearance" />
      </Harness>
    )

    expect(bandHeight(container)).toBe(0)
  })
})
