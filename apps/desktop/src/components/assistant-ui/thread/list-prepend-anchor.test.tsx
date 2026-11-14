import { AssistantRuntimeProvider, type ThreadMessage, useExternalStoreRuntime } from '@assistant-ui/react'
import { act, render } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { stubThreadEnvironment, stubThreadViewportSize } from '../test-utils'

import { Thread } from '.'

stubThreadEnvironment()
stubThreadViewportSize()

const CLIENT_H = 600
// A prepend grows the content: the backfill anchor is consumed only by a
// taller tree, so the fixture's scrollHeight must track what is mounted.
const GROUP_H = 400
let captureCount = Infinity
let lateHeight = 0

const scrollHeightNow = () =>
  lateHeight +
  (mountedGroups(window.document.body) > captureCount ? 100 : 0) +
  CLIENT_H +
  window.document.querySelectorAll('[data-slot="aui_message-group"]').length * GROUP_H

Object.defineProperty(HTMLElement.prototype, 'scrollHeight', {
  configurable: true,
  get: scrollHeightNow
})
Object.defineProperty(HTMLElement.prototype, 'clientHeight', {
  configurable: true,
  get: () => CLIENT_H
})

beforeEach(() => window.localStorage.clear())

async function settle(ticks = 20) {
  await act(async () => {
    for (let tick = 0; tick < ticks; tick += 1) {
      await new Promise<void>(resolve => window.setTimeout(resolve, 0))
    }
  })
}

// Heavy enough that the transcript outgrows RENDER_BUDGET (600 units) and the
// list hides older turns behind "Show earlier" — the shape of a branched
// session that inherits a long parent history.
const heavyText = 'x'.repeat(5000)
const createdAt = new Date('2026-08-01T00:00:00.000Z')

function heavyTranscript(turns: number): ThreadMessage[] {
  return Array.from({ length: turns }, (_, index) => [
    {
      id: `u-${index}`,
      role: 'user',
      content: [{ type: 'text', text: heavyText }],
      attachments: [],
      createdAt,
      metadata: { custom: {} }
    } as ThreadMessage,
    {
      id: `a-${index}`,
      role: 'assistant',
      content: [{ type: 'text', text: heavyText }],
      status: { type: 'complete', reason: 'stop' },
      createdAt,
      metadata: { unstable_state: null, unstable_annotations: [], unstable_data: [], steps: [], custom: {} }
    } as ThreadMessage
  ]).flat()
}

function Harness({ messages }: { messages: ThreadMessage[] }) {
  const runtime = useExternalStoreRuntime<ThreadMessage>({ isRunning: false, messages, onNew: async () => {} })

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <Thread sessionKey="long" />
    </AssistantRuntimeProvider>
  )
}

const mountedGroups = (container: HTMLElement) => container.querySelectorAll('[data-slot="aui_message-group"]').length

async function settledGroupCount(container: HTMLElement): Promise<number> {
  let count = mountedGroups(container)

  for (let round = 0; round < 20; round += 1) {
    await settle(10)
    const next = mountedGroups(container)

    if (next === count) {
      return count
    }

    count = next
  }

  return count
}

describe('visible-group prepend anchoring', () => {
  it.each([false, true, 'wheel', 'pointerdown', 'keydown', 'scroll'])(
    'preserves visual anchoring unless intent cancels it: %s',
    async delayed => {
      const previousObserver = globalThis.ResizeObserver
      const observers = new Set<{ callback: ResizeObserverCallback; targets: Set<Element> }>()
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
      let prefixShift = 0
      const { container } = render(<Harness messages={heavyTranscript(60)} />)
      const viewport = container.querySelector<HTMLElement>('[data-slot="aui_thread-viewport"]')!
      const windowed = await settledGroupCount(container)
      expect(windowed).toBeGreaterThan(0)
      expect(windowed).toBeLessThan(60)
      const originalRect = HTMLElement.prototype.getBoundingClientRect

      const rect = vi.spyOn(HTMLElement.prototype, 'getBoundingClientRect').mockImplementation(function (
        this: HTMLElement
      ) {
        if (this === viewport) {
          return new DOMRect(0, 32, 800, CLIENT_H)
        }

        if (this.getAttribute('data-slot') === 'aui_message-group' && viewport.contains(this)) {
          const index = [...viewport.querySelectorAll<HTMLElement>('[data-slot="aui_message-group"]')].indexOf(this)

          return new DOMRect(0, 32 + index * GROUP_H + prefixShift - viewport.scrollTop, 800, GROUP_H)
        }

        return originalRect.call(this)
      })

      try {
        viewport.scrollTop = scrollHeightNow() - CLIENT_H
        act(() => viewport.dispatchEvent(new Event('scroll')))
        viewport.scrollTop = 2000
        act(() => viewport.dispatchEvent(new Event('scroll')))
        act(() => viewport.dispatchEvent(new WheelEvent('wheel', { deltaY: -120 })))
        await settle(5)
        // At the clamped top, the next wheel (not a synthetic scroll event)
        // initiates paging. Capture before that event, with bottom follow escaped.
        viewport.scrollTop = 0
        const tracked = viewport.querySelector<HTMLElement>('[data-slot="aui_message-group"]')!
        const before = tracked.getBoundingClientRect().top
        expect(before).toBe(32)
        captureCount = windowed
        act(() => viewport.dispatchEvent(new WheelEvent('wheel', { deltaY: -120 })))
        await settle(2)
        expect(mountedGroups(container)).toBeGreaterThan(windowed)
        expect(viewport.contains(tracked)).toBe(true)
        expect(tracked.className).toContain('[content-visibility:auto]')
        expect(tracked.getBoundingClientRect().top).toBeCloseTo(before, 1)
        let inputTop: number | undefined

        if (delayed) {
          if (typeof delayed === 'string') {
            inputTop = viewport.scrollTop - 100

            if (delayed === 'scroll') {
              viewport.scrollTop = inputTop
              act(() => viewport.dispatchEvent(new Event('scroll')))
            } else {
              act(() =>
                viewport.dispatchEvent(
                  delayed === 'wheel' ? new WheelEvent('wheel', { deltaY: 120 }) : new Event(delayed)
                )
              )
              viewport.scrollTop = inputTop
            }
          }

          // A later content-visibility layout changes the prefix AFTER the
          // React commit. Deliver the actual component's resize observers.
          prefixShift = -156
          lateHeight = -156
          let delivered = 0
          act(() => {
            for (const observer of observers) {
              const targets = [...observer.targets].filter(el => el.getAttribute('data-slot') === 'aui_thread-content')

              if (targets.length) {
                delivered += 1
                observer.callback(
                  targets.map(target => ({
                    target,
                    contentRect: { height: scrollHeightNow() }
                  })) as ResizeObserverEntry[],
                  observer as unknown as ResizeObserver
                )
              }
            }
          })
          expect(delivered).toBeGreaterThan(0)

          if (inputTop === undefined) {
            expect(tracked.getBoundingClientRect().top).toBeCloseTo(before, 1)
          } else {
            expect(viewport.scrollTop).toBe(inputTop)
          }
        }

        await settle()

        if (inputTop === undefined) {
          expect(tracked.getBoundingClientRect().top).toBeCloseTo(before, 1)
        } else {
          expect(viewport.scrollTop).toBe(inputTop)
        }
      } finally {
        rect.mockRestore()
        captureCount = Infinity
        lateHeight = 0
        vi.stubGlobal('ResizeObserver', previousObserver)
      }
    }
  )
})
