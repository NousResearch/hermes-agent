import { act, cleanup, render } from '@testing-library/react'
import { StrictMode, useRef } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { COMPOSER_HEIGHT_VAR, COMPOSER_SURFACE_HEIGHT_VAR, THREAD_SETTLED_CLEARANCE_VAR } from '@/app/chat/surface-vars'

import { useComposerMetrics } from './use-composer-metrics'

let composerText = ''
let dockHeight = 320
let dockWidth = 480
let surfaceHeight = 72

vi.mock('@assistant-ui/react', () => ({
  useAuiState: (select: (state: { composer: { text: string } }) => unknown) =>
    select({ composer: { text: composerText } })
}))

// Keep upstream's real shared-observer lifecycle in the StrictMode proof.
const observers: FakeResizeObserver[] = []

class FakeResizeObserver implements ResizeObserver {
  readonly targets = new Set<Element>()

  constructor(readonly callback: ResizeObserverCallback) {
    observers.push(this)
  }

  disconnect() {
    this.targets.clear()
  }

  observe(target: Element) {
    this.targets.add(target)
  }

  unobserve(target: Element) {
    this.targets.delete(target)
  }
}

const syncMetrics = () => {
  for (const observer of observers) {
    const entries = [...observer.targets].map(target => ({ target }) as ResizeObserverEntry)

    if (entries.length > 0) {
      observer.callback(entries, observer)
    }
  }
}

const rect = (height: number, width: number): DOMRect =>
  ({ bottom: height, height, left: 0, right: width, toJSON: () => ({}), top: 0, width, x: 0, y: 0 }) as DOMRect

function MetricsHarness({
  poppedOut = false,
  running,
  surfaceKey = 'session-a'
}: {
  poppedOut?: boolean
  running: boolean
  surfaceKey?: string
}) {
  const composerDockRef = useRef<HTMLDivElement | null>(null)
  const composerRef = useRef<HTMLFormElement | null>(null)
  const composerSurfaceRef = useRef<HTMLDivElement | null>(null)
  const editorRef = useRef<HTMLDivElement | null>(null)

  const fit = useComposerMetrics({
    composerDockRef,
    composerRef,
    composerSurfaceRef,
    editorRef,
    poppedOut,
    running,
    surfaceKey
  })

  return (
    <div data-chat-surface>
      <div data-slot="composer-dock" ref={composerDockRef} />
      <form data-fit={JSON.stringify(fit)} data-slot="composer-root" ref={composerRef} />
      <div data-slot="composer-surface" ref={composerSurfaceRef} />
      <div data-slot="composer-editor" ref={editorRef} />
    </div>
  )
}

const renderHarness = (running: boolean) => render(<MetricsHarness running={running} />)

const surfaceOf = (container: HTMLElement) => container.querySelector<HTMLElement>('[data-chat-surface]')!

const mockComposerRects = () =>
  vi.spyOn(HTMLElement.prototype, 'getBoundingClientRect').mockImplementation(function (this: HTMLElement) {
    const slot = this.dataset.slot

    if (slot === 'composer-dock') {
      return rect(dockHeight, dockWidth)
    }

    if (slot === 'composer-root') {
      return rect(80, dockWidth)
    }

    if (slot === 'composer-surface') {
      return rect(surfaceHeight, 480)
    }

    return rect(28, 480)
  })

describe('useComposerMetrics', () => {
  beforeEach(() => {
    vi.stubGlobal('ResizeObserver', FakeResizeObserver)
  })

  afterEach(() => {
    cleanup()
    composerText = ''
    dockHeight = 320
    dockWidth = 480
    surfaceHeight = 72
    vi.restoreAllMocks()
    vi.unstubAllGlobals()
  })

  it('retains the running dock clearance when the turn settles, then releases it for the next run', () => {
    mockComposerRects()

    const view = renderHarness(true)
    const surface = surfaceOf(view.container)

    act(() => syncMetrics())
    expect(surface.style.getPropertyValue(COMPOSER_HEIGHT_VAR)).toBe('320px')

    dockHeight = 120
    view.rerender(<MetricsHarness running={false} />)
    act(() => syncMetrics())

    expect(surface.style.getPropertyValue(COMPOSER_HEIGHT_VAR)).toBe('120px')
    expect(surface.style.getPropertyValue(THREAD_SETTLED_CLEARANCE_VAR)).toBe('320px')

    view.rerender(<MetricsHarness running />)

    expect(surface.style.getPropertyValue(THREAD_SETTLED_CLEARANCE_VAR)).toBe('0px')
  })

  it('keeps the running floor active when status cleanup shrinks the dock before settlement', () => {
    mockComposerRects()

    const view = renderHarness(true)
    const surface = surfaceOf(view.container)

    act(() => syncMetrics())
    dockHeight = 120
    act(() => syncMetrics())

    expect(surface.style.getPropertyValue(COMPOSER_HEIGHT_VAR)).toBe('120px')
    expect(surface.style.getPropertyValue(THREAD_SETTLED_CLEARANCE_VAR)).toBe('320px')

    view.rerender(<MetricsHarness running={false} />)

    expect(surface.style.getPropertyValue(THREAD_SETTLED_CLEARANCE_VAR)).toBe('320px')
  })

  it('releases stale settled clearance when an idle hard newline expands without a resize delivery', () => {
    mockComposerRects()

    const view = renderHarness(true)
    const surface = surfaceOf(view.container)

    act(() => syncMetrics())
    dockHeight = 120
    view.rerender(<MetricsHarness running={false} />)
    act(() => syncMetrics())

    expect(surface.style.getPropertyValue(THREAD_SETTLED_CLEARANCE_VAR)).toBe('320px')

    dockHeight = 200
    composerText = 'first line\nsecond line'
    view.rerender(<MetricsHarness running={false} />)

    expect(surface.style.getPropertyValue(COMPOSER_HEIGHT_VAR)).toBe('200px')
    expect(surface.style.getPropertyValue(THREAD_SETTLED_CLEARANCE_VAR)).toBe('0px')
  })

  it('does not carry a settled clearance into another idle session', () => {
    mockComposerRects()

    const view = renderHarness(true)
    const surface = surfaceOf(view.container)

    act(() => syncMetrics())
    dockHeight = 120
    view.rerender(<MetricsHarness running={false} />)
    act(() => syncMetrics())

    expect(surface.style.getPropertyValue(THREAD_SETTLED_CLEARANCE_VAR)).toBe('320px')

    view.rerender(<MetricsHarness running={false} surfaceKey="session-b" />)

    expect(surface.style.getPropertyValue(THREAD_SETTLED_CLEARANCE_VAR)).toBe('0px')
  })

  it('republishes the dock height after StrictMode replays the cleanup', () => {
    dockHeight = 200
    surfaceHeight = 120
    mockComposerRects()

    const view = render(
      <StrictMode>
        <MetricsHarness running={false} />
      </StrictMode>
    )

    act(() => syncMetrics())

    const surface = surfaceOf(view.container)

    expect(surface.style.getPropertyValue(COMPOSER_HEIGHT_VAR)).toBe('200px')
    expect(surface.style.getPropertyValue(COMPOSER_SURFACE_HEIGHT_VAR)).toBe('120px')

    view.rerender(
      <StrictMode>
        <MetricsHarness running />
      </StrictMode>
    )
    dockHeight = 80
    act(() => syncMetrics())
    view.rerender(
      <StrictMode>
        <MetricsHarness running={false} />
      </StrictMode>
    )
    expect(surface.style.getPropertyValue(THREAD_SETTLED_CLEARANCE_VAR)).toBe('200px')

    view.unmount()
    expect(surface.style.getPropertyValue(COMPOSER_HEIGHT_VAR)).toBe('')
    expect(surface.style.getPropertyValue(COMPOSER_SURFACE_HEIGHT_VAR)).toBe('')
    expect(surface.style.getPropertyValue(THREAD_SETTLED_CLEARANCE_VAR)).toBe('')
  })

  it('skips the write when the bucketed height is unchanged within one mount', () => {
    dockHeight = 200
    surfaceHeight = 120
    mockComposerRects()

    const { container } = render(<MetricsHarness running={false} />)

    act(() => syncMetrics())

    const setProperty = vi.spyOn(surfaceOf(container).style, 'setProperty')

    act(() => syncMetrics())

    expect(setProperty).not.toHaveBeenCalled()
  })

  it('clears retained clearance while a floating composer still follows its progressive width fit', () => {
    mockComposerRects()

    const view = renderHarness(true)
    const surface = surfaceOf(view.container)

    dockHeight = 120
    view.rerender(<MetricsHarness running={false} />)
    expect(surface.style.getPropertyValue(THREAD_SETTLED_CLEARANCE_VAR)).toBe('320px')

    view.rerender(<MetricsHarness poppedOut running={false} />)
    expect(surface.style.getPropertyValue(COMPOSER_HEIGHT_VAR)).toBe('0px')
    expect(surface.style.getPropertyValue(COMPOSER_SURFACE_HEIGHT_VAR)).toBe('0px')
    expect(surface.style.getPropertyValue(THREAD_SETTLED_CLEARANCE_VAR)).toBe('0px')

    for (const width of [600, 480, 300, 240, 160, 600]) {
      dockWidth = width
      act(() => syncMetrics())
      expect(
        JSON.parse(view.container.querySelector<HTMLElement>('[data-slot="composer-root"]')!.dataset.fit!)
      ).toEqual({
        compactPill: width < 560,
        foldVoice: width < 260,
        minimal: width < 180,
        stacked: width < 320,
        tight: width < 320
      })
      expect(surface.style.getPropertyValue(THREAD_SETTLED_CLEARANCE_VAR)).toBe('0px')
    }

    view.rerender(<MetricsHarness running={false} />)
    expect(surface.style.getPropertyValue(COMPOSER_HEIGHT_VAR)).toBe('120px')
    expect(surface.style.getPropertyValue(THREAD_SETTLED_CLEARANCE_VAR)).toBe('0px')
  })
})
