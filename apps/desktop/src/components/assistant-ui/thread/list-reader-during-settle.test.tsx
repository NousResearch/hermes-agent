import { AssistantRuntimeProvider, type ThreadMessage, useExternalStoreRuntime } from '@assistant-ui/react'
import { act, render } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { rescopeConnectionScopedStores } from '@/lib/connection-scoped'
import { setActiveProfile } from '@/store/profile'

import { stubThreadEnvironment, stubThreadViewportSize } from '../test-utils'

import { Thread } from '.'

/**
 * The reported defect: the reader scrolls up while the transcript is still
 * settling (right after a switch, or while a turn re-arms the restore loop) and
 * the loop drags them back to the remembered target — the reader loses their
 * place. Reproduced in the isolated dev instance (900 px of drift with no user
 * input) before this test existed.
 */
const observers = new Set<{ fire: () => void }>()

class TestResizeObserver {
  private readonly callback: ResizeObserverCallback
  constructor(callback: ResizeObserverCallback) {
    this.callback = callback
    observers.add(this)
  }
  disconnect() {
    observers.delete(this)
  }
  fire() {
    this.callback(
      [
        {
          borderBoxSize: [],
          contentBoxSize: [],
          contentRect: {
            bottom: 1, height: 1, left: 0, right: 1, toJSON: () => ({}), top: 0, width: 1, x: 0, y: 0
          } as DOMRect,
          devicePixelContentBoxSize: [],
          target: document.body
        } as ResizeObserverEntry
      ],
      this as unknown as ResizeObserver
    )
  }
  observe() {}
  unobserve() {}
}

stubThreadEnvironment()
vi.stubGlobal('ResizeObserver', TestResizeObserver)
stubThreadViewportSize()

const SCROLL_H = 5000
const CLIENT_H = 600
const READER_FROM_BOTTOM = 900
const VIEWPORT_SLOT = 'aui_thread-viewport'
const CLEARANCE_SLOT = 'aui_composer-clearance'

let viewportScrollHeight = SCROLL_H
let viewportClientHeight = CLIENT_H
let clearanceHeight = 120

Object.defineProperty(HTMLElement.prototype, 'scrollHeight', {
  configurable: true,
  get() {
    return viewportScrollHeight
  }
})
Object.defineProperty(HTMLElement.prototype, 'clientHeight', {
  configurable: true,
  get() {
    if (this.getAttribute?.('data-slot') === CLEARANCE_SLOT) {
      return clearanceHeight
    }
    return viewportClientHeight
  }
})

beforeEach(() => {
  viewportScrollHeight = SCROLL_H
  viewportClientHeight = CLIENT_H
  clearanceHeight = 120
  observers.clear()
  window.localStorage.clear()
  setActiveProfile('default')
  rescopeConnectionScopedStores(null)
})

function viewportEl(container: HTMLElement): HTMLElement {
  const el = container.querySelector(`[data-slot="${VIEWPORT_SLOT}"]`) as HTMLElement | null
  expect(el).toBeTruthy()
  return el!
}

function fireContentResizes() {
  act(() => {
    for (const observer of [...observers]) {
      observer.fire()
    }
  })
}

async function settleTicks(ticks = 3) {
  await act(async () => {
    for (let tick = 0; tick < ticks; tick += 1) {
      await new Promise<void>(resolve => window.setTimeout(resolve, 0))
    }
  })
}

const createdAt = new Date('2026-08-01T00:00:00.000Z')

function messages(key: string, turns = 1): ThreadMessage[] {
  return Array.from({ length: turns }, (_, index) => [
    {
      id: `u-${key}-${index}`,
      role: 'user',
      content: [{ type: 'text', text: `message ${index} in ${key}` }],
      attachments: [],
      createdAt,
      metadata: { custom: {} }
    } as ThreadMessage,
    {
      id: `a-${key}-${index}`,
      role: 'assistant',
      content: [{ type: 'text', text: `response ${index} in ${key}` }],
      status: { type: 'complete', reason: 'stop' },
      createdAt,
      metadata: { unstable_state: null, unstable_annotations: [], unstable_data: [], steps: [], custom: {} }
    } as ThreadMessage
  ]).flat()
}

function Harness({ sessionKey }: { sessionKey: string }) {
  const runtime = useExternalStoreRuntime<ThreadMessage>({
    isRunning: false,
    messages: messages(sessionKey),
    onNew: async () => {}
  })

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <Thread clampToComposer sessionKey={sessionKey} />
    </AssistantRuntimeProvider>
  )
}

describe('a reader who scrolls during the settle window keeps their place', () => {
  it('does not drag the reader back to the bottom while the restore loop is still running', async () => {
    const { container } = render(<Harness sessionKey="reader" />)
    const vp = viewportEl(container)

    // The reader takes over MID-SETTLE: a foreign scroll position, reported like
    // the real thing (scrollTop write + scroll event) before the loop settles.
    const readerTop = SCROLL_H - CLIENT_H - READER_FROM_BOTTOM
    act(() => {
      vp.scrollTop = readerTop
      vp.dispatchEvent(new Event('scroll'))
    })

    // Content keeps arriving (the turn-end append) while the loop would re-apply.
    viewportScrollHeight = SCROLL_H + 2000
    fireContentResizes()
    await settleTicks(4)
    viewportScrollHeight = SCROLL_H + 2000
    fireContentResizes()

    expect(vp.scrollTop).toBe(readerTop)
  })
})
