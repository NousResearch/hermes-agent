import { AssistantRuntimeProvider, type ThreadMessage, useExternalStoreRuntime } from '@assistant-ui/react'
import { act, render } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { PaneLifecycleContext, PaneVisibleContext } from '@/components/pane-shell/pane-visibility'

import { rescopeConnectionScopedStores } from '@/lib/connection-scoped'
import { setActiveProfile } from '@/store/profile'
import {
  getThreadScrollPosition,
  requestScrollToBottom,
  saveThreadScrollPosition,
  threadScrollStorageKey
} from '@/store/thread-scroll'

import { stubThreadEnvironment, stubThreadViewportSize } from '../test-utils'

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

async function settleScroll(ticks = 3) {
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

interface ScrollHarnessProps {
  messages: ThreadMessage[]
  sessionKey: string | null
  scrollProfile?: string
  sessionId?: string | null
}

function ScrollHarness({ messages, sessionKey, scrollProfile, sessionId }: ScrollHarnessProps) {
  const runtime = useExternalStoreRuntime<ThreadMessage>({
    isRunning: false,
    messages,
    onNew: async () => {}
  })

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <Thread sessionKey={sessionKey} scrollProfile={scrollProfile} sessionId={sessionId} />
    </AssistantRuntimeProvider>
  )
}

describe('list session-scroll restore', () => {
  it('restores a reading offset on return after switching away', async () => {
    saveThreadScrollPosition('a', { fromBottom: 800, kind: 'offset' })

    const { container, rerender } = render(<ScrollHarness messages={sessionMessages('a')} sessionKey="a" />)
    const vp = viewportEl(container)

    await settleScroll()

    expect(vp.scrollTop).toBe(SCROLL_H - 800 - CLIENT_H)

    rerender(<ScrollHarness messages={sessionMessages('b')} sessionKey="b" />)
    const vpB = viewportEl(container)

    await settleScroll()

    expect(vpB.scrollTop).toBe(SCROLL_H - CLIENT_H)

    rerender(<ScrollHarness messages={sessionMessages('a')} sessionKey="a" />)
    const vpA = viewportEl(container)

    await settleScroll()

    expect(vpA.scrollTop).toBe(SCROLL_H - 800 - CLIENT_H)

    // Globals switch before React commits the outgoing cleanup.
    for (const remote of [null, 'https://other.invalid']) {
      setActiveProfile('other')
      rescopeConnectionScopedStores(remote ? { mode: 'remote', baseUrl: remote, profile: 'other' } : null)
      rerender(<ScrollHarness key={remote ?? 'profile'} messages={sessionMessages('a')} sessionKey="a" />)
      await settleScroll()
      expect(viewportEl(container).scrollTop).toBe(SCROLL_H - CLIENT_H)
    }
  })

  it.each([0, 800])('restores a kept-alive pane after hidden layout drift (offset %i)', async offset => {
    if (offset) saveThreadScrollPosition('a', { fromBottom: offset, kind: 'offset' })
    const messages = sessionMessages('a')
    const pane = (visible: boolean) => (
      <PaneVisibleContext.Provider value={visible}>
        <PaneLifecycleContext.Provider value={visible ? 'visible' : 'hot-hidden'}>
          <ScrollHarness messages={messages} sessionKey="a" />
        </PaneLifecycleContext.Provider>
      </PaneVisibleContext.Provider>
    )
    const { container, rerender } = render(pane(true))
    await settleScroll()
    const vp = viewportEl(container)
    expect(vp.scrollTop).toBe(SCROLL_H - CLIENT_H - offset)
    rerender(pane(false))
    // A hidden pane's smaller render budget / background refresh changes its
    // layout. These browser scroll events are not a new reading position.
    act(() => {
      vp.scrollTop = 300
      vp.dispatchEvent(new Event('scroll'))
    })
    await settleScroll()
    rerender(pane(true))
    await settleScroll()
    expect(vp.scrollTop).toBe(SCROLL_H - CLIENT_H - offset)
  })

  it.each([0, 800])('preserves position through repeated reveals with late resize (offset %i)', async offset => {
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
    if (offset) saveThreadScrollPosition('a', { fromBottom: offset, kind: 'offset' })
    const messages = sessionMessages('a')
    let runtimeId: string | null = null
    const pane = (visible: boolean) => (
      <PaneVisibleContext.Provider value={visible}>
        <PaneLifecycleContext.Provider value={visible ? 'visible' : 'hot-hidden'}>
          <ScrollHarness messages={messages} sessionKey="a" sessionId={runtimeId} />
        </PaneLifecycleContext.Provider>
      </PaneVisibleContext.Provider>
    )
    const { container, rerender, unmount } = render(pane(true))
    try {
      const vp = viewportEl(container)
      await settleScroll(10)
      for (let round = 0; round < 4; round++) {
        if (round === 2) {
          runtimeId = 'runtime-a'
          rerender(pane(true))
          await settleScroll(3)
        }
        // Deferred markdown finishes after the initial restore handed off.
        // Switch away in the same frame, before the library's queued follow.
        act(() => {
          scrollHeightValue += 1000
          for (const observer of observers) {
            const targets = [...observer.targets].filter(el => el.getAttribute('data-slot') === 'aui_thread-content')
            if (targets.length)
              observer.callback(
                targets.map(target => ({
                  target,
                  contentRect: { height: scrollHeightValue }
                })) as ResizeObserverEntry[],
                observer as unknown as ResizeObserver
              )
          }
        })
        rerender(pane(false))
        await settleScroll(10)
        rerender(pane(true))
        await settleScroll(10)
        expect(vp.scrollTop).toBeGreaterThanOrEqual(scrollHeightValue - CLIENT_H - offset - 1)
        expect(vp.scrollTop).toBeLessThanOrEqual(scrollHeightValue - CLIENT_H - offset)
      }
      // User input ends resize protection; a real reading position still wins.
      act(() => {
        vp.dispatchEvent(new Event('pointerdown'))
        vp.scrollTop -= 800
        vp.dispatchEvent(new Event('scroll'))
      })
      await settleScroll(10)
      const readingTop = vp.scrollTop
      rerender(pane(false))
      await settleScroll(10)
      rerender(pane(true))
      await settleScroll(10)
      expect(vp.scrollTop).toBe(readingTop)
      // Returning to the latest message must replace the old reading target,
      // including late resizes after the jump cancelled offset restoration.
      act(() => requestScrollToBottom(runtimeId))
      await settleScroll(10)
      expect(vp.scrollTop).toBeGreaterThanOrEqual(scrollHeightValue - CLIENT_H - 1)
      act(() => vp.dispatchEvent(new Event('scroll')))
      for (let round = 0; round < 3; round++) {
        act(() => {
          scrollHeightValue += 1000
          for (const observer of observers) {
            const targets = [...observer.targets].filter(el => el.getAttribute('data-slot') === 'aui_thread-content')
            if (targets.length)
              observer.callback(
                targets.map(target => ({
                  target,
                  contentRect: { height: scrollHeightValue }
                })) as ResizeObserverEntry[],
                observer as unknown as ResizeObserver
              )
          }
        })
        rerender(pane(false))
        await settleScroll(10)
        rerender(pane(true))
        await settleScroll(10)
        expect(vp.scrollTop).toBeGreaterThanOrEqual(scrollHeightValue - CLIENT_H - 1)
      }
    } finally {
      unmount()
      vi.stubGlobal('ResizeObserver', previousObserver)
    }
  })

  it.each([true, false])('keeps scroll ownership when profile switches before visibility: %s', async profileFirst => {
    saveThreadScrollPosition('a', { fromBottom: 800, kind: 'offset' })
    const ownerKey = threadScrollStorageKey()
    const messages = sessionMessages('a')
    const pane = (visible: boolean) => (
      <PaneVisibleContext.Provider value={visible}>
        <PaneLifecycleContext.Provider value={visible ? 'visible' : 'hot-hidden'}>
          <ScrollHarness messages={messages} sessionKey="a" />
        </PaneLifecycleContext.Provider>
      </PaneVisibleContext.Provider>
    )
    const { container, rerender } = render(pane(true))
    await settleScroll(10)
    const vp = viewportEl(container)
    if (profileFirst) setActiveProfile('pr-bot')
    rerender(pane(false))
    if (!profileFirst) setActiveProfile('pr-bot')
    const otherKey = threadScrollStorageKey()
    await settleScroll(10)
    // The selected global profile can still belong to the other Bot when the
    // default Bot's kept-alive pane reveals. Its transcript owner did not change.
    rerender(pane(true))
    await settleScroll(10)
    expect(vp.scrollTop).toBe(SCROLL_H - CLIENT_H - 800)
    act(() => requestScrollToBottom())
    await settleScroll(10)
    rerender(pane(false))
    await settleScroll(10)
    expect(getThreadScrollPosition('a', ownerKey)).toEqual({ kind: 'bottom' })
    expect(getThreadScrollPosition('a', otherKey)).toBeUndefined()
    act(() => window.dispatchEvent(new Event('beforeunload')))
    expect(getThreadScrollPosition('a', otherKey)).toBeUndefined()
    setActiveProfile('default')
    rerender(pane(true))
    await settleScroll(10)
    expect(vp.scrollTop).toBeGreaterThanOrEqual(SCROLL_H - CLIENT_H - 1)
  })

  it('restores remounted Bots from their explicit owners while a different profile is active', async () => {
    const defaultKey = threadScrollStorageKey('default')
    const otherKey = threadScrollStorageKey('pr-bot')
    saveThreadScrollPosition('default-chat', { fromBottom: 800, kind: 'offset' }, defaultKey)
    setActiveProfile('pr-bot')
    const pane = (bot: string, epoch = 0) => (
      <ScrollHarness
        key={`${bot}:${epoch}`}
        messages={sessionMessages(bot)}
        sessionKey={`${bot}-chat`}
        scrollProfile={bot}
      />
    )
    const { container, rerender } = render(pane('default'))
    await settleScroll(10)
    expect(viewportEl(container).scrollTop).toBe(SCROLL_H - CLIENT_H - 800)
    act(() => requestScrollToBottom())
    await settleScroll(10)
    act(() => window.dispatchEvent(new Event('beforeunload')))
    expect(getThreadScrollPosition('default-chat', defaultKey)).toEqual({ kind: 'bottom' })
    expect(getThreadScrollPosition('default-chat', otherKey)).toBeUndefined()
    for (let round = 0; round < 3; round++) {
      rerender(pane('pr-bot', round))
      await settleScroll(10)
      rerender(pane('default', round + 1))
      await settleScroll(10)
      expect(viewportEl(container).scrollTop).toBeGreaterThanOrEqual(SCROLL_H - CLIENT_H - 1)
    }
    expect(getThreadScrollPosition('default-chat', otherKey)).toBeUndefined()
  })

  it('re-arms restoration when a mounted transcript resolves a different owner profile', async () => {
    saveThreadScrollPosition('a', { fromBottom: 800, kind: 'offset' }, threadScrollStorageKey('default'))
    setActiveProfile('pr-bot')
    const messages = sessionMessages('a')
    const { container, rerender } = render(<ScrollHarness messages={messages} sessionKey="a" />)
    await settleScroll(10)
    rerender(<ScrollHarness messages={messages} sessionKey="a" scrollProfile="default" />)
    await settleScroll(10)
    expect(viewportEl(container).scrollTop).toBe(SCROLL_H - CLIENT_H - 800)
  })

  it('keeps bottom intent when content grows before a prepend and runtime binding', async () => {
    const messages = sessionMessages('a', 60).map(message => ({
      ...message,
      content: [{ type: 'text', text: 'x'.repeat(5000) }] as const
    }))
    const { container, rerender, getByText } = render(<ScrollHarness messages={messages} sessionKey="a" />)
    await settleScroll(20)
    const vp = viewportEl(container)
    expect(vp.scrollTop).toBeGreaterThanOrEqual(SCROLL_H - CLIENT_H - 1)
    // Layout grew before the library's next-frame bottom follow, matching a
    // delayed budget commit. A prepend must anchor the intended bottom.
    scrollHeightValue += 3700
    act(() => getByText('Show earlier messages').click())
    rerender(<ScrollHarness messages={messages} sessionKey="a" sessionId="runtime-a" />)
    await settleScroll(10)
    expect(vp.scrollTop).toBeGreaterThanOrEqual(scrollHeightValue - CLIENT_H - 1)
  })

  it('keeps a clamped cold offset parked until the transcript is tall enough', async () => {
    saveThreadScrollPosition('b', { fromBottom: 800, kind: 'offset' })
    scrollHeightValue = CLIENT_H

    const { container, rerender } = render(<ScrollHarness messages={[]} sessionKey="b" />)
    const vp = viewportEl(container)

    await settleScroll()

    scrollHeightValue = 1000
    rerender(<ScrollHarness messages={sessionMessages('b')} sessionKey="b" />)
    await settleScroll(20)

    expect(vp.scrollTop).toBe(0)

    rerender(<ScrollHarness messages={sessionMessages('b', 2)} sessionKey="b" />)
    await settleScroll()

    expect(vp.scrollTop).toBe(0)

    scrollHeightValue = 2000
    rerender(<ScrollHarness messages={sessionMessages('b', 3)} sessionKey="b" />)
    await settleScroll()

    expect(vp.scrollTop).toBe(2000 - CLIENT_H - 800)

    saveThreadScrollPosition('c', { fromBottom: 800, kind: 'offset' })
    scrollHeightValue = 1000
    rerender(<ScrollHarness messages={sessionMessages('c')} sessionKey="c" />)
    await settleScroll(20)
    act(() => vp.dispatchEvent(new WheelEvent('wheel', { deltaY: -200, bubbles: true })))
    scrollHeightValue = 2000
    rerender(<ScrollHarness messages={sessionMessages('c', 3)} sessionKey="c" />)
    await settleScroll()
    // jsdom has no native scroll anchoring; the browser probe checks the
    // resulting reader position. Here the abandoned target must not return.
    expect(vp.scrollTop).not.toBe(2000 - CLIENT_H - 800)
  })
})
