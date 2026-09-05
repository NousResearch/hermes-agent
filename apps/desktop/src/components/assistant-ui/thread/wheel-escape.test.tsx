// The transcript must release use-stick-to-bottom's lock when the reader
// scrolls up.
//
// The library's own wheel escape walks up from the wheel target until
// `getComputedStyle(el).overflow` is exactly 'scroll' or 'auto'. This viewport
// is `overflow-x-hidden overflow-y-auto`, which Chromium computes as
// 'hidden auto' — verified in real Chromium, control: a plain `overflow-y:auto`
// computes to 'auto' and would have matched. So the walk never reaches the
// viewport, the escape is dead code, and every content-growth tick of a
// streaming turn jammed a scrolled-up reader back to the newest message.
import { AssistantRuntimeProvider, type ThreadMessage, useExternalStoreRuntime } from '@assistant-ui/react'
import { cleanup, render } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { Thread } from '.'

const stopScroll = vi.fn()
const scrollToBottom = vi.fn()

vi.mock('use-stick-to-bottom', () => {
  const makeRef = () => {
    const ref = ((node: HTMLElement | null) => {
      ref.current = node
    }) as ((node: HTMLElement | null) => void) & { current: HTMLElement | null }

    ref.current = null

    return ref
  }

  const scrollRef = makeRef()
  const contentRef = makeRef()

  return {
    useStickToBottom: () => ({
      contentRef,
      escapedFromLock: false,
      isAtBottom: true,
      isNearBottom: true,
      scrollRef,
      scrollToBottom,
      state: {},
      stopScroll
    })
  }
})

class TestResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
}

vi.stubGlobal('ResizeObserver', TestResizeObserver)
vi.stubGlobal('requestAnimationFrame', (cb: FrameRequestCallback) => window.setTimeout(() => cb(performance.now()), 0))
vi.stubGlobal('cancelAnimationFrame', (id: number) => window.clearTimeout(id))
vi.stubGlobal('CSS', { escape: (s: string) => s })

Element.prototype.scrollTo = function scrollTo() {}

Element.prototype.animate = function animate() {
  return { cancel() {}, finished: Promise.resolve() } as unknown as Animation
}

const createdAt = new Date('2026-05-01T00:00:00.000Z')

const messages: ThreadMessage[] = [
  {
    attachments: [],
    content: [{ text: 'question', type: 'text' }],
    createdAt,
    id: 'u1',
    metadata: { custom: {} },
    role: 'user'
  } as ThreadMessage,
  {
    content: [{ text: 'a long streamed answer', type: 'text' }],
    createdAt,
    id: 'a1',
    metadata: { custom: {}, steps: [], unstable_annotations: [], unstable_data: [], unstable_state: null },
    role: 'assistant',
    status: { type: 'running' }
  } as ThreadMessage
]

function Harness() {
  const runtime = useExternalStoreRuntime<ThreadMessage>({ isRunning: true, messages, onNew: async () => {} })

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <Thread />
    </AssistantRuntimeProvider>
  )
}

/** jsdom reports every box as 0×0; the handler needs a scrollable viewport. */
function makeScrollable(el: HTMLElement, { client = 100, scroll = 900 } = {}) {
  Object.defineProperty(el, 'scrollHeight', { configurable: true, value: scroll })
  Object.defineProperty(el, 'clientHeight', { configurable: true, value: client })
}

function viewportOf(container: HTMLElement) {
  const el = container.querySelector<HTMLElement>('[data-slot="aui_thread-viewport"]')

  if (!el) {
    throw new Error('thread viewport not found')
  }

  return el
}

describe('transcript releases the stick-to-bottom lock on scroll-up', () => {
  beforeEach(() => {
    stopScroll.mockClear()
    scrollToBottom.mockClear()
  })

  afterEach(cleanup)

  it('escapes only for an upward wheel on a scrollable viewport', async () => {
    const { container, findByText } = render(<Harness />)

    await findByText('a long streamed answer')

    const viewport = viewportOf(container)

    makeScrollable(viewport)
    stopScroll.mockClear()

    viewport.dispatchEvent(new WheelEvent('wheel', { bubbles: true, deltaY: 120 }))

    expect(stopScroll).not.toHaveBeenCalled()

    viewport.dispatchEvent(new WheelEvent('wheel', { bubbles: true, deltaY: -120 }))

    expect(stopScroll).toHaveBeenCalled()
  })

  it('leaves the gesture to a nested scroller that still has room above', async () => {
    const { container, findByText } = render(<Harness />)

    await findByText('a long streamed answer')

    const viewport = viewportOf(container)

    makeScrollable(viewport)

    const nested = window.document.createElement('div')

    nested.style.overflowY = 'auto'
    Object.defineProperty(nested, 'scrollTop', { configurable: true, value: 40 })
    viewport.append(nested)
    stopScroll.mockClear()

    nested.dispatchEvent(new WheelEvent('wheel', { bubbles: true, deltaY: -120 }))

    expect(stopScroll).not.toHaveBeenCalled()

    Object.defineProperty(nested, 'scrollTop', { configurable: true, value: 0 })
    nested.dispatchEvent(new WheelEvent('wheel', { bubbles: true, deltaY: -120 }))

    expect(stopScroll).toHaveBeenCalled()
  })
})
