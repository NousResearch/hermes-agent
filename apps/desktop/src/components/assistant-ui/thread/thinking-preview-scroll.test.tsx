import { AssistantRuntimeProvider, type ThreadMessage, useExternalStoreRuntime } from '@assistant-ui/react'
import { act, render } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { stubThreadEnvironment, stubThreadViewportSize } from '../test-utils'

import { Thread } from '.'

stubThreadEnvironment()
stubThreadViewportSize()

// stubThreadEnvironment installs an inert observer; the pin effect needs one
// that actually delivers growth entries, like streaming.test.tsx.
const resizeObservers = new Set<TestResizeObserver>()

class TestResizeObserver {
  private target: Element | null = null

  constructor(private readonly callback: ResizeObserverCallback) {
    resizeObservers.add(this)
  }

  observe(target: Element) {
    this.target = target
  }

  unobserve() {}

  disconnect() {
    resizeObservers.delete(this)
  }

  trigger(height: number) {
    if (!this.target) {
      return
    }

    this.callback(
      [
        {
          borderBoxSize: [{ blockSize: height }] as unknown as ResizeObserverEntry['borderBoxSize'],
          contentRect: { height } as DOMRectReadOnly,
          target: this.target
        } as ResizeObserverEntry
      ],
      this as unknown as ResizeObserver
    )
  }
}

vi.stubGlobal('ResizeObserver', TestResizeObserver)

const createdAt = new Date('2026-09-01T00:00:00.000Z')

// The thinking preview body is a real scroller while streaming: jsdom has no
// layout, so the distances the pin/re-lock logic reads are stubbed on the
// prototype and moved per scenario below.
let scrollHeight = 1000
let clientHeight = 160
let observedHeight = 0

Object.defineProperty(HTMLElement.prototype, 'scrollHeight', {
  configurable: true,
  get: () => scrollHeight
})
Object.defineProperty(HTMLElement.prototype, 'clientHeight', {
  configurable: true,
  get: () => clientHeight
})

beforeEach(() => {
  window.localStorage.clear()
  resizeObservers.clear()
  scrollHeight = 1000
  clientHeight = 160
  observedHeight = 0
})

function streamingReasoningMessage(): ThreadMessage {
  return {
    id: 'assistant-reasoning-stream',
    role: 'assistant',
    content: [{ type: 'reasoning', text: ' Working through the problem step by step.', status: { type: 'running' } }],
    status: { type: 'running' },
    createdAt,
    metadata: { unstable_state: null, unstable_annotations: [], unstable_data: [], steps: [], custom: {} }
  } as unknown as ThreadMessage
}

function Harness() {
  const runtime = useExternalStoreRuntime<ThreadMessage>({
    messages: [streamingReasoningMessage()],
    isRunning: true,
    onNew: async () => {}
  })

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <Thread />
    </AssistantRuntimeProvider>
  )
}

function previewBody(container: HTMLElement): HTMLElement {
  const body = container.querySelector<HTMLElement>('[data-slot="aui_thinking-body"]')
  expect(body).toBeTruthy()

  return body!
}

function growthFrame() {
  // Streamed tokens land: the observer reports a taller content box and the
  // pin handler runs with the grown scrollHeight already visible to it.
  scrollHeight += 200
  observedHeight += 100

  for (const observer of resizeObservers) {
    observer.trigger(observedHeight)
  }
}

async function settle(ticks = 8) {
  await act(async () => {
    for (let tick = 0; tick < ticks; tick += 1) {
      await new Promise<void>(resolve => window.setTimeout(resolve, 0))
    }
  })
}

describe('streaming thinking preview scroll', () => {
  it('pins the live preview to its bottom while the reader has not scrolled away', async () => {
    const { container } = render(<Harness />)
    await settle()

    const body = previewBody(container)
    expect(body.className).toContain('max-h-40')
    expect(body.className).not.toMatch(/overscroll-contain/)

    // Reading at the bottom: growth keeps the newest tokens visible.
    body.scrollTop = scrollHeight - clientHeight
    await act(async () => {
      growthFrame()
    })
    // jsdom performs no layout, so the assigned value is not clamped.
    expect(body.scrollTop).toBe(scrollHeight)
  })

  it('pauses the pin while the reader has scrolled the preview up', async () => {
    const { container } = render(<Harness />)
    await settle()

    const body = previewBody(container)

    // The reader scrolls up inside the streaming preview.
    body.scrollTop = 120
    await act(async () => {
      body.dispatchEvent(new Event('scroll'))
    })

    const held = body.scrollTop
    await act(async () => {
      growthFrame()
    })
    // Growth no longer yanks the reader back down.
    expect(body.scrollTop).toBe(held)
  })

  it('re-locks the pin once the reader returns to the bottom', async () => {
    const { container } = render(<Harness />)
    await settle()

    const body = previewBody(container)

    body.scrollTop = 120
    await act(async () => {
      body.dispatchEvent(new Event('scroll'))
    })
    await act(async () => {
      growthFrame()
    })
    expect(body.scrollTop).toBe(120)

    // Back to the bottom: follow resumes on the next growth.
    body.scrollTop = scrollHeight - clientHeight
    await act(async () => {
      body.dispatchEvent(new Event('scroll'))
    })
    await act(async () => {
      growthFrame()
    })
    expect(body.scrollTop).toBe(scrollHeight)
  })
})
