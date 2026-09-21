import { AssistantRuntimeProvider, type ThreadMessage, useExternalStoreRuntime } from '@assistant-ui/react'
import { act, render, waitFor } from '@testing-library/react'
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

import { TranscriptWindowProvider, type TranscriptWindowValue } from './transcript-window'

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
  isRunning?: boolean
  messages: ThreadMessage[]
  sessionKey: string | null
  scrollProfile?: string
  sessionId?: string | null
  window?: TranscriptWindowValue
}

function ScrollHarness({
  isRunning = false,
  messages,
  sessionKey,
  scrollProfile,
  sessionId,
  window
}: ScrollHarnessProps) {
  const runtime = useExternalStoreRuntime<ThreadMessage>({
    isRunning,
    messages,
    onNew: async () => {}
  })

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <TranscriptWindowProvider value={window ?? { olderAvailable: false, expandWindow: () => {} }}>
        <Thread scrollProfile={scrollProfile} sessionId={sessionId} sessionKey={sessionKey} />
      </TranscriptWindowProvider>
    </AssistantRuntimeProvider>
  )
}

describe('list session-scroll restore', () => {
  it('follows streaming content growth before paint while the reader is pinned', async () => {
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

    const { container, unmount } = render(
      <ScrollHarness isRunning messages={sessionMessages('stream-follow')} sessionKey="stream-follow" />
    )

    try {
      const vp = viewportEl(container)
      await settleScroll(10)
      expect(vp.scrollTop).toBeGreaterThanOrEqual(scrollHeightValue - CLIENT_H - 1)

      const deliverContentResize = () => {
        for (const observer of observers) {
          const targets = [...observer.targets].filter(el => el.getAttribute('data-slot') === 'aui_thread-content')

          if (targets.length) {
            observer.callback(
              targets.map(target => ({
                target,
                contentRect: { height: scrollHeightValue }
              })) as ResizeObserverEntry[],
              observer as unknown as ResizeObserver
            )
          }
        }
      }

      act(deliverContentResize)

      const raf = vi.spyOn(globalThis, 'requestAnimationFrame').mockImplementation(() => 0)

      try {
        act(() => {
          scrollHeightValue += 168
          deliverContentResize()
        })

        expect(vp.scrollTop).toBe(scrollHeightValue - CLIENT_H)
      } finally {
        raf.mockRestore()
      }
    } finally {
      unmount()
      vi.stubGlobal('ResizeObserver', previousObserver)
    }
  })

  it('lets a reader escape bottom-follow when a running transcript grows during the scroll gesture', async () => {
    // #116273: a pending clarify keeps the turn running while the transcript
    // can still resize. If that resize lands in the same frame as scroll-up,
    // the reader's intent must win over bottom-follow.
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

    const messages = sessionMessages('pending')
    const { container, unmount } = render(<ScrollHarness isRunning messages={messages} sessionKey="pending" />)

    try {
      const vp = viewportEl(container)
      await settleScroll(10)

      const deliverContentResize = (height: number) => {
        for (const observer of observers) {
          const targets = [...observer.targets].filter(el => el.getAttribute('data-slot') === 'aui_thread-content')

          if (targets.length) {
            observer.callback(
              targets.map(target => ({ target, contentRect: { height } })) as ResizeObserverEntry[],
              observer as unknown as ResizeObserver
            )
          }
        }
      }

      act(() => deliverContentResize(scrollHeightValue))
      await settleScroll(3)
      expect(vp.scrollTop).toBeGreaterThanOrEqual(scrollHeightValue - CLIENT_H - 1)

      const readingTop = scrollHeightValue - CLIENT_H - 900

      act(() => {
        vp.dispatchEvent(new WheelEvent('wheel', { bubbles: true, deltaY: -160 }))
        vp.scrollTop = readingTop
        vp.dispatchEvent(new Event('scroll'))
        scrollHeightValue += 120
        deliverContentResize(scrollHeightValue)
      })

      await settleScroll(10)

      expect(vp.scrollTop).toBeLessThan(scrollHeightValue - CLIENT_H - 100)
      expect(vp.scrollTop).toBeGreaterThanOrEqual(readingTop - 1)
    } finally {
      unmount()
      vi.stubGlobal('ResizeObserver', previousObserver)
    }
  })

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
    if (offset) {
      saveThreadScrollPosition('a', { fromBottom: offset, kind: 'offset' })
    }

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

    if (offset) {
      saveThreadScrollPosition('a', { fromBottom: offset, kind: 'offset' })
    }

    const messages = sessionMessages('a')
    let runtimeId: string | null = null

    const pane = (visible: boolean) => (
      <PaneVisibleContext.Provider value={visible}>
        <PaneLifecycleContext.Provider value={visible ? 'visible' : 'hot-hidden'}>
          <ScrollHarness messages={messages} sessionId={runtimeId} sessionKey="a" />
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

            if (targets.length) {
              observer.callback(
                targets.map(target => ({
                  target,
                  contentRect: { height: scrollHeightValue }
                })) as ResizeObserverEntry[],
                observer as unknown as ResizeObserver
              )
            }
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

            if (targets.length) {
              observer.callback(
                targets.map(target => ({
                  target,
                  contentRect: { height: scrollHeightValue }
                })) as ResizeObserverEntry[],
                observer as unknown as ResizeObserver
              )
            }
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

    if (profileFirst) {
      setActiveProfile('pr-bot')
    }

    rerender(pane(false))

    if (!profileFirst) {
      setActiveProfile('pr-bot')
    }

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
        scrollProfile={bot}
        sessionKey={`${bot}-chat`}
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
    rerender(<ScrollHarness messages={messages} scrollProfile="default" sessionKey="a" />)
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
    rerender(<ScrollHarness messages={messages} sessionId="runtime-a" sessionKey="a" />)
    await settleScroll(10)
    expect(vp.scrollTop).toBeGreaterThanOrEqual(scrollHeightValue - CLIENT_H - 1)
  })

  it('keeps a clamped cold offset parked until the transcript is tall enough', async () => {
    const pendingHistory = { olderAvailable: true, expandWindow: () => new Promise<boolean>(() => {}) }
    saveThreadScrollPosition('b', { fromBottom: 800, kind: 'offset' })
    scrollHeightValue = CLIENT_H

    const { container, rerender } = render(<ScrollHarness messages={[]} sessionKey="b" window={pendingHistory} />)
    const vp = viewportEl(container)

    await settleScroll()

    scrollHeightValue = 1000
    rerender(<ScrollHarness messages={sessionMessages('b')} sessionKey="b" window={pendingHistory} />)
    await settleScroll(20)

    expect(vp.scrollTop).toBe(0)

    rerender(<ScrollHarness messages={sessionMessages('b', 2)} sessionKey="b" window={pendingHistory} />)
    await settleScroll()

    expect(vp.scrollTop).toBe(0)

    scrollHeightValue = 2000
    rerender(<ScrollHarness messages={sessionMessages('b', 3)} sessionKey="b" window={pendingHistory} />)
    await settleScroll()

    expect(vp.scrollTop).toBe(2000 - CLIENT_H - 800)

    saveThreadScrollPosition('c', { fromBottom: 800, kind: 'offset' })
    scrollHeightValue = 1000
    rerender(<ScrollHarness messages={sessionMessages('c')} sessionKey="c" window={pendingHistory} />)
    await settleScroll(20)
    act(() => vp.dispatchEvent(new WheelEvent('wheel', { deltaY: -200, bubbles: true })))
    scrollHeightValue = 2000
    rerender(<ScrollHarness messages={sessionMessages('c', 3)} sessionKey="c" window={pendingHistory} />)
    await settleScroll()
    // jsdom has no native scroll anchoring; the browser probe checks the
    // resulting reader position. Here the abandoned target must not return.
    expect(vp.scrollTop).not.toBe(2000 - CLIENT_H - 800)
  })

  it('applies the load target in the commit that prepends rows, before another animation frame', async () => {
    const raf = vi.spyOn(globalThis, 'requestAnimationFrame').mockImplementation(() => 0)

    try {
      const { container, rerender } = render(<ScrollHarness messages={sessionMessages('commit')} sessionKey="commit" />)
      const vp = viewportEl(container)
      expect(vp.scrollTop).toBe(SCROLL_H - CLIENT_H)
      scrollHeightValue += 2000
      rerender(<ScrollHarness messages={sessionMessages('commit', 2)} sessionKey="commit" />)
      // Runtime publication is timer-batched; animation frames remain frozen.
      await waitFor(() => expect(container.querySelectorAll('[data-slot="aui_message-group"]').length).toBe(2))
      expect(vp.scrollTop).toBe(scrollHeightValue - CLIENT_H)
    } finally {
      raf.mockRestore()
    }
  })

  it.each(['exhausted', 'failed'] as const)('releases an unreachable restore when history is %s', async outcome => {
    saveThreadScrollPosition('lost', { fromBottom: 9000, kind: 'offset' })
    const expandWindow = vi.fn(async () => false)
    const history = { olderAvailable: outcome === 'failed', expandWindow }

    const { container, getByText } = render(
      <ScrollHarness messages={sessionMessages('lost')} sessionKey="lost" window={history} />
    )

    // Separate acts let the stepped budget and parked-target effects commit.
    for (let step = 0; step < 5; step++) {
      await settleScroll(20)
    }

    act(() => window.dispatchEvent(new Event('beforeunload')))
    expect(getThreadScrollPosition('lost')).toEqual({ kind: 'offset', fromBottom: SCROLL_H - CLIENT_H })
    expect(viewportEl(container).scrollTop).toBe(0)

    if (outcome === 'failed') {
      expect(expandWindow).toHaveBeenCalledTimes(1)
      act(() => getByText('Show earlier messages').click())
      await settleScroll(10)
      expect(expandWindow).toHaveBeenCalledTimes(2)
    }
  })

  it('lets downward wheel intent cancel a clamped restore before delayed history arrives', async () => {
    saveThreadScrollPosition('cancel', { fromBottom: 9000, kind: 'offset' })
    const history = { olderAvailable: true, expandWindow: () => new Promise<boolean>(() => {}) }

    const { container, rerender } = render(
      <ScrollHarness messages={sessionMessages('cancel')} sessionKey="cancel" window={history} />
    )

    await settleScroll(20)
    const vp = viewportEl(container)
    act(() => vp.dispatchEvent(new WheelEvent('wheel', { deltaY: 200, bubbles: true })))
    scrollHeightValue = 12000
    rerender(<ScrollHarness messages={sessionMessages('cancel', 2)} sessionKey="cancel" window={history} />)
    await settleScroll(10)
    expect(vp.scrollTop).not.toBe(12000 - CLIENT_H - 9000)
  })

  it('captures the current reader at async page arrival and ignores an outgoing page after switching', async () => {
    let finish!: (applied: boolean) => void
    let capture: (() => void) | undefined

    const expandWindow = vi.fn((before?: () => void) => {
      capture = before

      return new Promise<boolean>(resolve => {
        finish = resolve
      })
    })

    const history = { olderAvailable: true, expandWindow }

    const { container, rerender, getByText } = render(
      <ScrollHarness messages={sessionMessages('arrival')} sessionKey="arrival" window={history} />
    )

    await settleScroll(10)
    const vp = viewportEl(container)
    act(() => {
      vp.dispatchEvent(new Event('pointerdown'))
      vp.scrollTop = 300
      vp.dispatchEvent(new Event('scroll'))
      getByText('Show earlier messages').click()
    })
    await settleScroll(10)
    act(() => {
      vp.scrollTop = 800
      vp.dispatchEvent(new Event('scroll'))
      capture?.()
      scrollHeightValue += 2000
      rerender(<ScrollHarness messages={sessionMessages('arrival', 2)} sessionKey="arrival" window={history} />)
      finish(true)
    })
    await settleScroll(10)
    expect(vp.scrollTop).toBe(2800)
    act(() => getByText('Show earlier messages').click())
    rerender(<ScrollHarness messages={sessionMessages('next')} sessionKey="next" />)
    await settleScroll(10)
    const nextTop = vp.scrollTop
    act(() => {
      capture?.()
      finish(true)
    })
    await settleScroll(10)
    expect(vp.scrollTop).toBe(nextTop)
  })

  it('waits for a history page before requesting another or spending the restore budget', async () => {
    saveThreadScrollPosition('async', { fromBottom: 9000, kind: 'offset' })
    let finish!: (applied: boolean) => void
    let beforePrepend: (() => void) | undefined

    const expandWindow = vi.fn((capture?: () => void) => {
      beforePrepend = capture

      return new Promise<boolean>(resolve => {
        finish = resolve
      })
    })

    const window = { olderAvailable: true, expandWindow }

    const { container, rerender } = render(
      <ScrollHarness messages={sessionMessages('async')} sessionKey="async" window={window} />
    )

    await waitFor(() => expect(expandWindow).toHaveBeenCalled())
    expect(expandWindow).toHaveBeenCalledTimes(1)
    await settleScroll(120)
    expect(expandWindow).toHaveBeenCalledTimes(1)
    expect(viewportEl(container).scrollTop).toBe(0)
    act(() => {
      beforePrepend?.()
      finish(true)
    })
    // RPC completion can precede the runtime's batched publication.
    await settleScroll(20)
    expect(expandWindow).toHaveBeenCalledTimes(1)
    act(() => {
      scrollHeightValue = 12000
      rerender(<ScrollHarness messages={sessionMessages('async', 3)} sessionKey="async" window={window} />)
      finish(true)
    })
    await settleScroll(20)
    expect(viewportEl(container).scrollTop).toBe(12000 - CLIENT_H - 9000)
    expect(expandWindow).toHaveBeenCalledTimes(1)
  })

  it('spends pages to regrow toward a saved offset the mounted window cannot cover', async () => {
    // Heavy turns so the transcript is windowed by the render budget; a saved
    // offset far below the mounted window must auto-spend pages (the same
    // Show-earlier path) instead of stranding at the clamped top.
    const heavy = 'x'.repeat(5000)

    const messages = Array.from({ length: 60 }, (_, i) => [
      {
        id: `u-${i}`,
        role: 'user',
        content: [{ type: 'text', text: heavy }],
        attachments: [],
        createdAt,
        metadata: { custom: {} }
      } as ThreadMessage,
      {
        id: `a-${i}`,
        role: 'assistant',
        content: [{ type: 'text', text: heavy }],
        status: { type: 'complete', reason: 'stop' },
        createdAt,
        metadata: { unstable_state: null, unstable_annotations: [], unstable_data: [], steps: [], custom: {} }
      } as ThreadMessage
    ]).flat()

    // Control: no saved offset — the default window stays capped.
    const ctl = render(<ScrollHarness messages={messages} sessionKey="ctl" />)

    const stableGroups = async (container: HTMLElement) => {
      let count = container.querySelectorAll('[data-slot="aui_message-group"]').length

      for (let round = 0; round < 25; round += 1) {
        await settleScroll(10)
        const next = container.querySelectorAll('[data-slot="aui_message-group"]').length

        if (next === count) {
          return count
        }

        count = next
      }

      return count
    }

    const ctlGroups = await stableGroups(ctl.container)
    ctl.unmount()

    // Deep saved offset — the regrow effect spends pages toward it.
    saveThreadScrollPosition('regrow', { fromBottom: 999999, kind: 'offset' })
    const deep = render(<ScrollHarness messages={messages} sessionKey="regrow" />)
    const deepGroups = await stableGroups(deep.container)

    expect(ctlGroups).toBeGreaterThan(0)
    expect(deepGroups).toBeGreaterThan(ctlGroups)
  })
})
