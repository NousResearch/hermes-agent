import { act, cleanup, render } from '@testing-library/react'
import { StrictMode, useRef } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import {
  COMPOSER_HEIGHT_VAR,
  COMPOSER_SURFACE_HEIGHT_VAR,
  THREAD_SETTLED_CLEARANCE_VAR
} from '@/app/chat/surface-vars'

import { useComposerMetrics } from './use-composer-metrics'

let composerText = ''
let dockHeight = 320
let surfaceHeight = 72
let syncMetrics: (() => void) | undefined

vi.mock('@assistant-ui/react', () => ({
  useAuiState: (select: (state: { composer: { text: string } }) => unknown) => select({ composer: { text: composerText } })
}))

vi.mock('@/hooks/use-resize-observer', () => ({
  useResizeObserver: (callback: () => void) => {
    syncMetrics = callback
  }
}))

const rect = (height: number, width: number): DOMRect =>
  ({ bottom: height, height, left: 0, right: width, toJSON: () => ({}), top: 0, width, x: 0, y: 0 }) as DOMRect

function MetricsHarness({ running, surfaceKey = 'session-a' }: { running: boolean; surfaceKey?: string }) {
  const composerDockRef = useRef<HTMLDivElement | null>(null)
  const composerRef = useRef<HTMLFormElement | null>(null)
  const composerSurfaceRef = useRef<HTMLDivElement | null>(null)
  const editorRef = useRef<HTMLDivElement | null>(null)

  useComposerMetrics({
    composerDockRef,
    composerRef,
    composerSurfaceRef,
    editorRef,
    poppedOut: false,
    running,
    surfaceKey
  })

  return (
    <div data-chat-surface>
      <div data-slot="composer-dock" ref={composerDockRef} />
      <form data-slot="composer-root" ref={composerRef} />
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
      return rect(dockHeight, 480)
    }

    if (slot === 'composer-root') {
      return rect(80, 480)
    }

    if (slot === 'composer-surface') {
      return rect(surfaceHeight, 480)
    }

    return rect(28, 480)
  })

describe('useComposerMetrics', () => {
  afterEach(() => {
    cleanup()
    composerText = ''
    dockHeight = 320
    surfaceHeight = 72
    syncMetrics = undefined
    vi.restoreAllMocks()
  })

  it('retains the running dock clearance when the turn settles, then releases it for the next run', () => {
    mockComposerRects()

    const view = renderHarness(true)
    const surface = surfaceOf(view.container)

    act(() => syncMetrics?.())
    expect(surface.style.getPropertyValue(COMPOSER_HEIGHT_VAR)).toBe('320px')

    dockHeight = 120
    view.rerender(<MetricsHarness running={false} />)
    act(() => syncMetrics?.())

    expect(surface.style.getPropertyValue(COMPOSER_HEIGHT_VAR)).toBe('120px')
    expect(surface.style.getPropertyValue(THREAD_SETTLED_CLEARANCE_VAR)).toBe('320px')

    view.rerender(<MetricsHarness running />)

    expect(surface.style.getPropertyValue(THREAD_SETTLED_CLEARANCE_VAR)).toBe('0px')
  })

  it('keeps the running floor active when status cleanup shrinks the dock before settlement', () => {
    mockComposerRects()

    const view = renderHarness(true)
    const surface = surfaceOf(view.container)

    act(() => syncMetrics?.())
    dockHeight = 120
    act(() => syncMetrics?.())

    expect(surface.style.getPropertyValue(COMPOSER_HEIGHT_VAR)).toBe('120px')
    expect(surface.style.getPropertyValue(THREAD_SETTLED_CLEARANCE_VAR)).toBe('320px')

    view.rerender(<MetricsHarness running={false} />)

    expect(surface.style.getPropertyValue(THREAD_SETTLED_CLEARANCE_VAR)).toBe('320px')
  })

  it('releases stale settled clearance when an idle hard newline expands without a resize delivery', () => {
    mockComposerRects()

    const view = renderHarness(true)
    const surface = surfaceOf(view.container)

    act(() => syncMetrics?.())
    dockHeight = 120
    view.rerender(<MetricsHarness running={false} />)
    act(() => syncMetrics?.())

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

    act(() => syncMetrics?.())
    dockHeight = 120
    view.rerender(<MetricsHarness running={false} />)
    act(() => syncMetrics?.())

    expect(surface.style.getPropertyValue(THREAD_SETTLED_CLEARANCE_VAR)).toBe('320px')

    view.rerender(<MetricsHarness running={false} surfaceKey="session-b" />)

    expect(surface.style.getPropertyValue(THREAD_SETTLED_CLEARANCE_VAR)).toBe('0px')
  })

  it('republishes the dock height after StrictMode replays the cleanup', () => {
    dockHeight = 200
    surfaceHeight = 120
    mockComposerRects()

    const { container } = render(
      <StrictMode>
        <MetricsHarness running={false} />
      </StrictMode>
    )

    act(() => syncMetrics?.())

    const surface = surfaceOf(container)

    expect(surface.style.getPropertyValue(COMPOSER_HEIGHT_VAR)).toBe('200px')
    expect(surface.style.getPropertyValue(COMPOSER_SURFACE_HEIGHT_VAR)).toBe('120px')
  })

  it('skips the write when the bucketed height is unchanged within one mount', () => {
    dockHeight = 200
    surfaceHeight = 120
    mockComposerRects()

    const { container } = render(<MetricsHarness running={false} />)

    act(() => syncMetrics?.())

    const setProperty = vi.spyOn(surfaceOf(container).style, 'setProperty')

    act(() => syncMetrics?.())

    expect(setProperty).not.toHaveBeenCalled()
  })
})
