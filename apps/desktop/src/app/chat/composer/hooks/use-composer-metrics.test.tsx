import { act, cleanup, render } from '@testing-library/react'
import { type RefObject, StrictMode, useRef } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { COMPOSER_HEIGHT_VAR, COMPOSER_SURFACE_HEIGHT_VAR } from '@/app/chat/surface-vars'

import { useComposerMetrics } from './use-composer-metrics'

vi.mock('@assistant-ui/react', () => ({
  useAuiState: (selector: (state: { composer: { text: string } }) => unknown) =>
    selector({ composer: { text: composerText } })
}))

let composerText = ''

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

const deliverResize = () =>
  act(() => {
    for (const observer of observers) {
      const entries = [...observer.targets].map(target => ({ target }) as ResizeObserverEntry)

      if (entries.length > 0) {
        observer.callback(entries, observer)
      }
    }
  })

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
  editorScrollHeight = 0,
  surfaceHeight
}: {
  dockHeight: number
  editorScrollHeight?: number | ((stacked: boolean) => number)
  surfaceHeight: number
}) {
  const composerDockRef = useRef<HTMLDivElement | null>(null)
  const composerRef = useRef<HTMLFormElement | null>(null)
  const composerSurfaceRef = useRef<HTMLDivElement | null>(null)
  const editorRef = useRef<HTMLDivElement | null>(null)

  const metrics = useComposerMetrics({ composerDockRef, composerRef, composerSurfaceRef, editorRef, poppedOut: false })

  return (
    <div data-chat-surface="">
      <div ref={sized(composerDockRef, dockHeight)}>
        <form ref={composerRef}>
          <div ref={sized(composerSurfaceRef, surfaceHeight)}>
            <div
              data-stacked={metrics.stacked}
              ref={node => {
                if (node) {
                  Object.defineProperty(node, 'scrollHeight', {
                    configurable: true,
                    get: () =>
                      typeof editorScrollHeight === 'number'
                        ? editorScrollHeight
                        : editorScrollHeight(node.dataset.stacked === 'true')
                  })
                }

                editorRef.current = node
              }}
            />
          </div>
        </form>
      </div>
    </div>
  )
}

const surfaceOf = (container: HTMLElement) => container.querySelector<HTMLElement>('[data-chat-surface]')!

describe('useComposerMetrics', () => {
  beforeEach(() => {
    // The hook retains its shared observer across tests; keep its delivery route.
    composerText = ''
    vi.stubGlobal('ResizeObserver', FakeResizeObserver)
  })

  afterEach(() => {
    cleanup()
    vi.unstubAllGlobals()
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

  it('keeps the stacked layout when a tall resting editor is emptied', () => {
    composerText = 'two lines'
    const view = render(<Harness dockHeight={200} editorScrollHeight={78} surfaceHeight={120} />)

    deliverResize()
    expect(view.container.querySelector('[data-stacked="true"]')).not.toBeNull()

    composerText = ''
    view.rerender(<Harness dockHeight={200} editorScrollHeight={78} surfaceHeight={120} />)

    deliverResize()
    expect(view.container.querySelector('[data-stacked="true"]')).not.toBeNull()
  })

  it('keeps a nonempty draft stacked across wrap-boundary resize deliveries', () => {
    composerText = 'a draft that wraps in the inline row'

    const view = render(
      <Harness dockHeight={200} editorScrollHeight={stacked => (stacked ? 28 : 56)} surfaceHeight={120} />
    )

    expect(view.container.querySelector('[data-stacked="true"]')).not.toBeNull()

    for (let delivery = 0; delivery < 3; delivery++) {
      deliverResize()
      expect(view.container.querySelector('[data-stacked="true"]')).not.toBeNull()
    }
  })

  it('collapses a default-height editor when the draft is emptied without a resize', () => {
    composerText = 'two lines'
    const view = render(<Harness dockHeight={200} editorScrollHeight={56} surfaceHeight={120} />)

    expect(view.container.querySelector('[data-stacked="true"]')).not.toBeNull()

    composerText = ''
    view.rerender(<Harness dockHeight={200} editorScrollHeight={28} surfaceHeight={120} />)

    expect(view.container.querySelector('[data-stacked="false"]')).not.toBeNull()
    deliverResize()
    expect(view.container.querySelector('[data-stacked="false"]')).not.toBeNull()
  })
})
