import { act, render } from '@testing-library/react'
import { useRef } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { useComposerMetrics } from './use-composer-metrics'

let composerText = ''
let dockHeight = 320
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

describe('useComposerMetrics', () => {
  afterEach(() => {
    composerText = ''
    dockHeight = 320
    syncMetrics = undefined
    vi.restoreAllMocks()
  })

  it('retains the running dock clearance when the turn settles, then releases it for the next run', () => {
    vi.spyOn(HTMLElement.prototype, 'getBoundingClientRect').mockImplementation(function (this: HTMLElement) {
      const slot = this.dataset.slot

      if (slot === 'composer-dock') {
        return rect(dockHeight, 480)
      }

      if (slot === 'composer-root') {
        return rect(80, 480)
      }

      if (slot === 'composer-surface') {
        return rect(72, 480)
      }

      return rect(28, 480)
    })

    const view = renderHarness(true)
    const surface = view.container.querySelector<HTMLElement>('[data-chat-surface]')

    act(() => syncMetrics?.())
    expect(surface?.style.getPropertyValue('--composer-measured-height')).toBe('320px')

    dockHeight = 120
    view.rerender(<MetricsHarness running={false} />)
    act(() => syncMetrics?.())

    expect(surface?.style.getPropertyValue('--composer-measured-height')).toBe('120px')
    expect(surface?.style.getPropertyValue('--thread-settled-clearance-height')).toBe('320px')

    view.rerender(<MetricsHarness running />)

    expect(surface?.style.getPropertyValue('--thread-settled-clearance-height')).toBe('0px')
  })

  it('keeps the running floor active when status cleanup shrinks the dock before settlement', () => {
    vi.spyOn(HTMLElement.prototype, 'getBoundingClientRect').mockImplementation(function (this: HTMLElement) {
      return rect(this.dataset.slot === 'composer-dock' ? dockHeight : 72, 480)
    })

    const view = renderHarness(true)
    const surface = view.container.querySelector<HTMLElement>('[data-chat-surface]')

    act(() => syncMetrics?.())
    dockHeight = 120
    act(() => syncMetrics?.())

    expect(surface?.style.getPropertyValue('--composer-measured-height')).toBe('120px')
    expect(surface?.style.getPropertyValue('--thread-settled-clearance-height')).toBe('320px')

    view.rerender(<MetricsHarness running={false} />)

    expect(surface?.style.getPropertyValue('--thread-settled-clearance-height')).toBe('320px')
  })

  it('releases stale settled clearance when an idle hard newline expands without a resize delivery', () => {
    vi.spyOn(HTMLElement.prototype, 'getBoundingClientRect').mockImplementation(function (this: HTMLElement) {
      return rect(this.dataset.slot === 'composer-dock' ? dockHeight : 72, 480)
    })

    const view = renderHarness(true)
    const surface = view.container.querySelector<HTMLElement>('[data-chat-surface]')

    act(() => syncMetrics?.())
    dockHeight = 120
    view.rerender(<MetricsHarness running={false} />)
    act(() => syncMetrics?.())

    expect(surface?.style.getPropertyValue('--thread-settled-clearance-height')).toBe('320px')

    dockHeight = 200
    composerText = 'first line\nsecond line'
    view.rerender(<MetricsHarness running={false} />)

    expect(surface?.style.getPropertyValue('--composer-measured-height')).toBe('200px')
    expect(surface?.style.getPropertyValue('--thread-settled-clearance-height')).toBe('0px')
  })

  it('does not carry a settled clearance into another idle session', () => {
    vi.spyOn(HTMLElement.prototype, 'getBoundingClientRect').mockImplementation(function (this: HTMLElement) {
      return rect(this.dataset.slot === 'composer-dock' ? dockHeight : 72, 480)
    })

    const view = renderHarness(true)
    const surface = view.container.querySelector<HTMLElement>('[data-chat-surface]')

    act(() => syncMetrics?.())
    dockHeight = 120
    view.rerender(<MetricsHarness running={false} />)
    act(() => syncMetrics?.())

    expect(surface?.style.getPropertyValue('--thread-settled-clearance-height')).toBe('320px')

    view.rerender(<MetricsHarness running={false} surfaceKey="session-b" />)

    expect(surface?.style.getPropertyValue('--thread-settled-clearance-height')).toBe('0px')
  })
})
