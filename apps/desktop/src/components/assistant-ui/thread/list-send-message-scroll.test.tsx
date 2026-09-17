import { AssistantRuntimeProvider, type ThreadMessage, useExternalStoreRuntime } from '@assistant-ui/react'
import { act, render } from '@testing-library/react'
import { beforeEach, describe, expect, it } from 'vitest'

import { rescopeConnectionScopedStores } from '@/lib/connection-scoped'
import { setActiveProfile } from '@/store/profile'

import { stubThreadEnvironment, stubThreadViewportSize } from '../test-utils'

import { Thread } from '.'

// A faithful-enough viewport: the browser CLAMPS scrollTop into
// [0, scrollHeight - clientHeight] whenever layout shrinks, and raises a
// 'scroll' event for that clamp. jsdom does neither, and both are load-bearing
// for the send-message path this file covers.
const observers = new Set<{ fire: (height: number) => void; targets: Set<Element> }>()

class TestResizeObserver {
  readonly targets = new Set<Element>()

  constructor(private readonly callback: ResizeObserverCallback) {
    observers.add(this)
  }

  disconnect() {
    this.targets.clear()
    observers.delete(this)
  }

  fire(height: number) {
    const targets = [...this.targets]

    if (targets.length === 0) {
      return
    }

    this.callback(
      targets.map(target => ({ contentRect: { height } as DOMRect, target })) as ResizeObserverEntry[],
      this as unknown as ResizeObserver
    )
  }

  observe(target: Element) {
    this.targets.add(target)
  }

  unobserve(target: Element) {
    this.targets.delete(target)
  }
}

stubThreadEnvironment()
stubThreadViewportSize()

const SCROLL_H = 5000
const CLIENT_H = 600
const CLEARANCE_H = 120
const VIEWPORT_SLOT = 'aui_thread-viewport'
const CLEARANCE_SLOT = 'aui_composer-clearance'

let viewportScrollHeight = SCROLL_H
let viewportClientHeight = CLIENT_H
let clearanceHeight = CLEARANCE_H

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

const scrollTops = new WeakMap<HTMLElement, number>()

Object.defineProperty(HTMLElement.prototype, 'scrollTop', {
  configurable: true,
  get() {
    return scrollTops.get(this as HTMLElement) ?? 0
  },
  set(value: number) {
    const max = Math.max(0, viewportScrollHeight - viewportClientHeight)
    scrollTops.set(this as HTMLElement, Math.min(Math.max(0, value), max))
  }
})

beforeEach(() => {
  viewportScrollHeight = SCROLL_H
  viewportClientHeight = CLIENT_H
  clearanceHeight = CLEARANCE_H
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

const bottomTop = () => Math.max(0, viewportScrollHeight - viewportClientHeight)

function fireContentResizes() {
  act(() => {
    for (const observer of [...observers]) {
      observer.fire(viewportScrollHeight)
    }
  })
}

/** Re-lay out the viewport the way the browser does: clamp scrollTop into the
 *  new range and raise the resulting 'scroll' event, then deliver the resize. */
function relayout(vp: HTMLElement, next: { clearance?: number; clientHeight?: number; scrollHeight?: number }) {
  act(() => {
    const before = vp.scrollTop
    viewportScrollHeight = next.scrollHeight ?? viewportScrollHeight
    viewportClientHeight = next.clientHeight ?? viewportClientHeight
    clearanceHeight = next.clearance ?? clearanceHeight

    const clamped = Math.min(before, bottomTop())

    if (clamped !== before) {
      vp.scrollTop = clamped
      vp.dispatchEvent(new Event('scroll'))
    }
  })

  fireContentResizes()
}

async function settleScroll(ticks = 3) {
  await act(async () => {
    for (let tick = 0; tick < ticks; tick += 1) {
      await new Promise<void>(resolve => window.setTimeout(resolve, 0))
    }
  })
}

const createdAt = new Date('2026-08-01T00:00:00.000Z')

function userMessage(id: string, text: string): ThreadMessage {
  return {
    id,
    role: 'user',
    content: [{ type: 'text', text }],
    attachments: [],
    createdAt,
    metadata: { custom: {} }
  } as ThreadMessage
}

function assistantMessage(id: string, text: string): ThreadMessage {
  return {
    id,
    role: 'assistant',
    content: [{ type: 'text', text }],
    status: { type: 'complete', reason: 'stop' },
    createdAt,
    metadata: { unstable_state: null, unstable_annotations: [], unstable_data: [], steps: [], custom: {} }
  } as ThreadMessage
}

const settledTurn = [userMessage('u-1', 'first question'), assistantMessage('a-1', 'first answer')]
const sentTurn = [...settledTurn, userMessage('u-2', 'second question')]

function SendHarness({ isRunning, messages }: { isRunning: boolean; messages: ThreadMessage[] }) {
  const runtime = useExternalStoreRuntime<ThreadMessage>({
    isRunning,
    messages,
    onNew: async () => {}
  })

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <Thread clampToComposer sessionKey="send" />
    </AssistantRuntimeProvider>
  )
}

describe('list scroll on sending a message', () => {
  it('rests flush at the bottom after a submit collapses the composer and appends a turn', async () => {
    const { container, rerender } = render(<SendHarness isRunning={false} messages={settledTurn} />)
    const vp = viewportEl(container)

    await settleScroll(10)
    fireContentResizes()
    await settleScroll(5)

    // Parked at the bottom, restore settled, before the user sends anything.
    expect(vp.scrollTop).toBe(bottomTop())

    // Submit: the composer clears and collapses back to one line, so the
    // clearance spacer shrinks and the clamped viewport box grows. Transcript
    // rows did not move — this is the composer-only resize the restore RO
    // deliberately ignores — but the browser still clamps scrollTop and fires
    // a 'scroll' for it.
    relayout(vp, { clearance: 60, clientHeight: CLIENT_H + 60, scrollHeight: SCROLL_H - 60 })

    // The new user message + the running assistant placeholder commit, and the
    // transcript grows by their height.
    act(() => {
      viewportScrollHeight = SCROLL_H - 60 + 420
    })
    rerender(<SendHarness isRunning messages={sentTurn} />)
    await settleScroll(10)
    fireContentResizes()
    await settleScroll(10)

    // Flush at the new bottom. Anything smaller is the reported symptom: the
    // transcript resting some pixels above the latest turn after a send.
    expect(vp.scrollTop).toBe(bottomTop())
  })
})
