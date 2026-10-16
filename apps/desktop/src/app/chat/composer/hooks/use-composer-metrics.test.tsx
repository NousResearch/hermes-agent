import { cleanup, render } from '@testing-library/react'
import { type RefObject, StrictMode, useRef } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { COMPOSER_HEIGHT_VAR, COMPOSER_SURFACE_HEIGHT_VAR } from '@/app/chat/surface-vars'

import { COMPOSER_MINIMAL_PX } from '../composer-utils'

import { useComposerMetrics } from './use-composer-metrics'

vi.mock('@assistant-ui/react', () => ({
  useAuiState: (selector: (state: { composer: { text: string } }) => unknown) => selector({ composer: { text: '' } })
}))

// The shared ResizeObserver delivers once per observed element after
// observe(); replay that by hand so the hook's initial measurement runs where
// it would in Chromium.
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

const deliverResize = () => {
  for (const observer of observers) {
    const entries = [...observer.targets].map(target => ({ target }) as ResizeObserverEntry)

    if (entries.length > 0) {
      observer.callback(entries, observer)
    }
  }
}

const sized =
  (ref: RefObject<HTMLDivElement | null>, height: number) =>
  (node: HTMLDivElement | null): void => {
    if (node) {
      node.getBoundingClientRect = () =>
        ({ bottom: height, height, left: 0, right: 640, top: 0, width: 640, x: 0, y: 0, toJSON: () => ({}) }) as DOMRect
    }

    ref.current = node
  }

function Harness({
  dockHeight,
  surfaceHeight,
  width = 640
}: {
  dockHeight: number
  surfaceHeight: number
  width?: number
}) {
  const composerDockRef = useRef<HTMLDivElement | null>(null)
  const composerRef = useRef<HTMLFormElement | null>(null)
  const composerSurfaceRef = useRef<HTMLDivElement | null>(null)
  const editorRef = useRef<HTMLDivElement | null>(null)

  const fit = useComposerMetrics({ composerDockRef, composerRef, composerSurfaceRef, editorRef, poppedOut: false })

  return (
    <div data-chat-surface="">
      <div ref={sized(composerDockRef, dockHeight)}>
        <form
          ref={node => {
            composerRef.current = node

            if (node) {
              node.getBoundingClientRect = () => ({ width, height: surfaceHeight }) as DOMRect
            }
          }}
        >
          <output data-testid="fit">{JSON.stringify(fit)}</output>
          <div ref={sized(composerSurfaceRef, surfaceHeight)}>
            <div ref={editorRef} />
          </div>
        </form>
      </div>
    </div>
  )
}

const surfaceOf = (container: HTMLElement) => container.querySelector<HTMLElement>('[data-chat-surface]')!

describe('useComposerMetrics — published clearance survives an effect replay', () => {
  beforeEach(() => {
    observers.length = 0
    vi.stubGlobal('ResizeObserver', FakeResizeObserver)
  })

  afterEach(() => {
    cleanup()
    vi.unstubAllGlobals()
  })

  it('uses a single control column below the two-touch-target budget', () => {
    vi.spyOn(window, 'getComputedStyle').mockReturnValue({ getPropertyValue: () => '44px' } as unknown as CSSStyleDeclaration)
    const { getByTestId } = render(<Harness dockHeight={200} surfaceHeight={120} width={80} />)
    expect(JSON.parse(getByTestId('fit').textContent!)).toMatchObject({ minimal: true, singleColumn: true })
    vi.restoreAllMocks()
  })

  it('collapses extra controls earlier when touch targets consume more width', () => {
    const minimalAt = (controlPx: number) => {
      const style = vi.spyOn(window, 'getComputedStyle').mockReturnValue({
        getPropertyValue: (name: string) => (name === '--composer-control-size' ? `${controlPx}px` : '')
      } as unknown as CSSStyleDeclaration)

      // Just past the point where desktop-size controls still fit.
      const { getByTestId, unmount } = render(
        <Harness dockHeight={200} surfaceHeight={120} width={COMPOSER_MINIMAL_PX + 1} />
      )

      const { minimal } = JSON.parse(getByTestId('fit').textContent!) as { minimal: boolean }
      unmount()
      style.mockRestore()

      return minimal
    }

    expect(minimalAt(24)).toBe(false)
    expect(minimalAt(44)).toBe(true)
  })

  it('republishes the dock height after StrictMode replays the cleanup', () => {
    // StrictMode replays every effect: mount → cleanup → mount. The cleanup
    // clears the surface vars; the replayed measurement must write them back
    // even though the dock's size has not changed since the first write.
    const { container } = render(
      <StrictMode>
        <Harness dockHeight={200} surfaceHeight={120} />
      </StrictMode>
    )

    deliverResize()

    const surface = surfaceOf(container)

    expect(surface.style.getPropertyValue(COMPOSER_HEIGHT_VAR)).toBe('200px')
    expect(surface.style.getPropertyValue(COMPOSER_SURFACE_HEIGHT_VAR)).toBe('120px')
  })
})
