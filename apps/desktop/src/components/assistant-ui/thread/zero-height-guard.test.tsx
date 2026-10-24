import { act, render } from '@testing-library/react'
import { useRef } from 'react'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import { useZeroHeightViewportGuard } from './zero-height-guard'

/**
 * Stands in for the browser's ResizeObserver so a test can deliver the exact
 * geometry jsdom cannot compute. The hook rides the app's ONE shared observer,
 * so every render lands on the same instance and `observe` re-points it at the
 * current test's element.
 */
class ReportResizeObserver {
  static instances: ReportResizeObserver[] = []

  static primary(): ReportResizeObserver {
    const existing = ReportResizeObserver.instances[0]

    if (existing) {
      return existing
    }

    throw new Error('no ResizeObserver was constructed yet')
  }

  callback: ResizeObserverCallback
  target: Element | null = null

  constructor(callback: ResizeObserverCallback) {
    this.callback = callback
    ReportResizeObserver.instances.push(this)
  }

  disconnect() {}

  emit(height: number) {
    this.callback(
      [{ contentRect: { height, width: 640 } as DOMRectReadOnly, target: this.target } as ResizeObserverEntry],
      this as unknown as ResizeObserver
    )
  }

  observe(target: Element) {
    this.target = target
  }

  unobserve() {}
}

/** What the guard's own fresh read sees when it probes. */
let measuredHeight = 0

function Viewport({ armed, onCollapse }: { armed: boolean; onCollapse: () => void }) {
  const ref = useRef<HTMLDivElement>(null)

  useZeroHeightViewportGuard({ armed, onCollapse, viewportRef: ref })

  return <div data-testid="viewport" ref={ref} />
}

/** Let the one-frame probe land. */
async function settle() {
  await act(async () => {
    await new Promise(resolve => setTimeout(resolve, 40))
  })
}

beforeAll(() => {
  vi.stubGlobal('ResizeObserver', ReportResizeObserver)
  vi.spyOn(Element.prototype, 'getBoundingClientRect').mockImplementation(
    () =>
      ({
        bottom: measuredHeight,
        height: measuredHeight,
        left: 0,
        right: 640,
        toJSON: () => ({}),
        top: 0,
        width: 640,
        x: 0,
        y: 0
      }) as DOMRect
  )
})

beforeEach(() => {
  measuredHeight = 0
})

afterEach(() => {
  delete (globalThis.document as { visibilityState?: string }).visibilityState
})

describe('useZeroHeightViewportGuard', () => {
  it('reports a viewport that survives the re-measure at zero height', async () => {
    const onCollapse = vi.fn()

    render(<Viewport armed onCollapse={onCollapse} />)

    act(() => ReportResizeObserver.primary().emit(0))
    await settle()

    expect(onCollapse).toHaveBeenCalledTimes(1)
  })

  it('stays quiet when the pane is hidden or the transcript is empty', async () => {
    const onCollapse = vi.fn()

    render(<Viewport armed={false} onCollapse={onCollapse} />)

    act(() => ReportResizeObserver.primary().emit(0))
    await settle()

    expect(onCollapse).not.toHaveBeenCalled()
  })

  it('stays quiet when the resize nudge restores the height', async () => {
    const onCollapse = vi.fn()

    render(<Viewport armed onCollapse={onCollapse} />)

    act(() => ReportResizeObserver.primary().emit(0))
    // The nudge worked: the probe's own read now finds a real box.
    measuredHeight = 812
    await settle()

    expect(onCollapse).not.toHaveBeenCalled()
  })

  it('asks the pane shell to re-measure before reporting', () => {
    const onCollapse = vi.fn()
    const onResize = vi.fn()

    window.addEventListener('resize', onResize)
    render(<Viewport armed onCollapse={onCollapse} />)

    act(() => ReportResizeObserver.primary().emit(0))

    expect(onResize).toHaveBeenCalledTimes(1)
    window.removeEventListener('resize', onResize)
  })

  it('reports once per episode, no matter how many zero deliveries arrive', async () => {
    const onCollapse = vi.fn()

    render(<Viewport armed onCollapse={onCollapse} />)

    act(() => ReportResizeObserver.primary().emit(0))
    await settle()
    act(() => ReportResizeObserver.primary().emit(0))
    act(() => ReportResizeObserver.primary().emit(0))
    await settle()

    expect(onCollapse).toHaveBeenCalledTimes(1)
  })

  it('re-arms after a healthy delivery, so a second collapse reports again', async () => {
    const onCollapse = vi.fn()

    render(<Viewport armed onCollapse={onCollapse} />)

    act(() => ReportResizeObserver.primary().emit(0))
    await settle()
    act(() => ReportResizeObserver.primary().emit(640))
    act(() => ReportResizeObserver.primary().emit(0))
    await settle()

    expect(onCollapse).toHaveBeenCalledTimes(2)
  })

  it('never claims a collapse from an empty delivery', async () => {
    const onCollapse = vi.fn()

    render(<Viewport armed onCollapse={onCollapse} />)

    act(() => ReportResizeObserver.primary().callback([], ReportResizeObserver.primary() as unknown as ResizeObserver))
    await settle()

    expect(onCollapse).not.toHaveBeenCalled()
  })

  it('ignores a minimized window, where every pane measures zero', async () => {
    const onCollapse = vi.fn()

    Object.defineProperty(globalThis.document, 'visibilityState', { configurable: true, value: 'hidden' })
    render(<Viewport armed onCollapse={onCollapse} />)

    act(() => ReportResizeObserver.primary().emit(0))
    await settle()

    expect(onCollapse).not.toHaveBeenCalled()
  })
})
