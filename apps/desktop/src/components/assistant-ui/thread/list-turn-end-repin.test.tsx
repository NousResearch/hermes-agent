import { AssistantRuntimeProvider, type ThreadMessage, useExternalStoreRuntime } from '@assistant-ui/react'
import { act, render } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { PaneLifecycleContext, PaneVisibleContext } from '@/components/pane-shell/pane-visibility'
import { rescopeConnectionScopedStores } from '@/lib/connection-scoped'
import { setActiveProfile } from '@/store/profile'

import { stubThreadEnvironment, stubThreadViewportSize } from '../test-utils'

import { TranscriptWindowProvider } from './transcript-window'

import { Thread } from '.'

stubThreadEnvironment()
stubThreadViewportSize()

const SCROLL_H = 5000
const CLIENT_H = 600
let scrollHeightValue = SCROLL_H

Object.defineProperty(HTMLElement.prototype, 'scrollHeight', {
  configurable: true,
  get() {
    return scrollHeightValue
  }
})
Object.defineProperty(HTMLElement.prototype, 'clientHeight', {
  configurable: true,
  get() {
    return CLIENT_H
  }
})

beforeEach(() => {
  scrollHeightValue = SCROLL_H
  window.localStorage.clear()
  setActiveProfile('default')
  rescopeConnectionScopedStores(null)
})

const VIEWPORT_SLOT = 'aui_thread-viewport'

function viewportEl(container: HTMLElement): HTMLElement {
  const el = container.querySelector(`[data-slot="${VIEWPORT_SLOT}"]`) as HTMLElement | null
  expect(el).toBeTruthy()

  return el!
}

async function settleScroll(ticks = 6) {
  await act(async () => {
    for (let tick = 0; tick < ticks; tick += 1) {
      await new Promise<void>(resolve => window.setTimeout(resolve, 0))
    }
  })
}

const createdAt = new Date('2026-08-01T00:00:00.000Z')

function sessionMessages(key: string, turns = 1): ThreadMessage[] {
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

interface TurnEndHarnessProps {
  isRunning: boolean
  messages: ThreadMessage[]
  sessionKey: string
}

function TurnEndHarness({ isRunning, messages, sessionKey }: TurnEndHarnessProps) {
  const runtime = useExternalStoreRuntime<ThreadMessage>({
    isRunning,
    messages,
    onNew: async () => {}
  })

  return (
    <PaneVisibleContext.Provider value>
      <PaneLifecycleContext.Provider value="visible">
        <AssistantRuntimeProvider runtime={runtime}>
          <TranscriptWindowProvider value={{ olderAvailable: false, expandWindow: () => {} }}>
            <Thread clampToComposer={false} sessionKey={sessionKey} />
          </TranscriptWindowProvider>
        </AssistantRuntimeProvider>
      </PaneLifecycleContext.Provider>
    </PaneVisibleContext.Provider>
  )
}

function stubResizeObservers() {
  const previousObserver = globalThis.ResizeObserver

  const observers = new Set<{
    callback: ResizeObserverCallback
    targets: Set<Element>
  }>()

  vi.stubGlobal(
    'ResizeObserver',
    class {
      targets = new Set<Element>()
      constructor(public callback: ResizeObserverCallback) {
        observers.add(this)
      }
      observe(target: Element) {
        this.targets.add(target)
      }
      unobserve(target: Element) {
        this.targets.delete(target)
      }
      disconnect() {
        this.targets.clear()
      }
    }
  )

  const deliverContentResize = () => {
    act(() => {
      for (const observer of observers) {
        const targets = [...observer.targets].filter(el => el.getAttribute('data-slot') === 'aui_thread-content')

        if (targets.length) {
          observer.callback(
            targets.map(target => ({ target, contentRect: { height: scrollHeightValue } })) as ResizeObserverEntry[],
            observer as unknown as ResizeObserver
          )
        }
      }
    })
  }

  return {
    deliverContentResize,
    restore() {
      vi.stubGlobal('ResizeObserver', previousObserver)
    }
  }
}

describe('list turn-end scroll re-pin', () => {
  it('re-pins a bottom-band reader when the finishing turn churns the layout (#135583)', async () => {
    const { deliverContentResize, restore } = stubResizeObservers()
    const messages = sessionMessages('turn-end')

    try {
      const { container, rerender } = render(<TurnEndHarness isRunning messages={messages} sessionKey="turn-end" />)

      const vp = viewportEl(container)

      await settleScroll()
      // The restore settled the reader at the bottom of the running turn.
      expect(vp.scrollTop).toBeGreaterThanOrEqual(SCROLL_H - CLIENT_H - 1)

      // A mouse-wheel reader's habitual notches (any wheel event) cancel the
      // restore ResizeObserver, so it cannot act as a fallback here — exactly
      // the packaged-build report: wheel users lose the reply, trackpad users
      // who never wheel keep the restore leg connected and never see it.
      act(() => {
        vp.dispatchEvent(new WheelEvent('wheel', { bubbles: true, deltaY: 120 }))
      })
      await settleScroll(3)

      // Turn end: the loading indicator unmounts (content shrinks) and the
      // browser clamps scrollTop down to the new maximum. jsdom has no layout,
      // so the clamp is written by hand: first a scroll event at the
      // pre-clamp position seeds the library's lastScrollTop (jsdom never
      // synthesizes scroll history), then the clamped write reads as a
      // scroll-up — exactly the misread a real clamp produces.
      act(() => {
        vp.scrollTop = SCROLL_H - CLIENT_H
        vp.dispatchEvent(new Event('scroll'))
      })
      scrollHeightValue -= 80
      act(() => {
        vp.scrollTop = scrollHeightValue - CLIENT_H
        vp.dispatchEvent(new Event('scroll'))
      })
      await settleScroll(3)

      // Deferred markdown settles after the run flag drops: the content grows
      // back past its pre-clamp height. The library has escaped (the clamp read
      // as a scroll-up), so its follow leg no-ops; only the end-of-turn re-pin
      // can bring the finished reply back into view.
      const clampedTop = vp.scrollTop
      scrollHeightValue += 300

      rerender(<TurnEndHarness isRunning={false} messages={messages} sessionKey="turn-end" />)
      // Let the runtime publish isRunning=false through its store before the
      // resize lands — in a real browser the layout churn follows the flag.
      await settleScroll(3)
      deliverContentResize()
      await settleScroll()

      expect(clampedTop).toBe(SCROLL_H - 80 - CLIENT_H)
      expect(vp.scrollTop).toBeGreaterThanOrEqual(scrollHeightValue - CLIENT_H - 1)
    } finally {
      restore()
    }
  })

  it('keeps a reader who scrolled up during the run parked at their position (#135583)', async () => {
    const { deliverContentResize, restore } = stubResizeObservers()
    const messages = sessionMessages('history-read')

    try {
      const { container, rerender } = render(<TurnEndHarness isRunning messages={messages} sessionKey="history-read" />)

      const vp = viewportEl(container)

      await settleScroll()

      // The reader wheels up into history mid-run: far past the snap band.
      const readingTop = SCROLL_H - CLIENT_H - 900

      act(() => {
        vp.dispatchEvent(new WheelEvent('wheel', { bubbles: true, deltaY: -160 }))
        vp.scrollTop = readingTop
        vp.dispatchEvent(new Event('scroll'))
      })
      await settleScroll(3)

      rerender(<TurnEndHarness isRunning={false} messages={messages} sessionKey="history-read" />)
      scrollHeightValue += 300
      deliverContentResize()
      await settleScroll()

      expect(vp.scrollTop).toBe(readingTop)
    } finally {
      restore()
    }
  })

  it('does not re-pin a reader who was already in history when the turn started', async () => {
    const { deliverContentResize, restore } = stubResizeObservers()
    const messages = sessionMessages('parked-reader')

    try {
      const { container, rerender } = render(
        <TurnEndHarness isRunning={false} messages={messages} sessionKey="parked-reader" />
      )

      const vp = viewportEl(container)

      await settleScroll()

      const readingTop = SCROLL_H - CLIENT_H - 900

      act(() => {
        vp.dispatchEvent(new WheelEvent('wheel', { bubbles: true, deltaY: -160 }))
        vp.scrollTop = readingTop
        vp.dispatchEvent(new Event('scroll'))
      })
      await settleScroll(3)

      // A new run starts (isRunning flips) while the reader is up in history.
      rerender(<TurnEndHarness isRunning messages={messages} sessionKey="parked-reader" />)
      await settleScroll(3)

      rerender(<TurnEndHarness isRunning={false} messages={messages} sessionKey="parked-reader" />)
      scrollHeightValue += 300
      deliverContentResize()
      await settleScroll()

      expect(vp.scrollTop).toBe(readingTop)
    } finally {
      restore()
    }
  })
})
