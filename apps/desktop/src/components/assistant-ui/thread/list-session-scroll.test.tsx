import { AssistantRuntimeProvider, type ThreadMessage, useExternalStoreRuntime } from '@assistant-ui/react'
import { act, render } from '@testing-library/react'
import { beforeEach, describe, expect, it } from 'vitest'

import { rescopeConnectionScopedStores } from '@/lib/connection-scoped'
import { setActiveProfile } from '@/store/profile'
import { getThreadScrollPosition, saveThreadScrollPosition, threadScrollStorageKey } from '@/store/thread-scroll'

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
  profile?: string
  sessionKey: string | null
}

function ScrollHarness({ messages, profile, sessionKey }: ScrollHarnessProps) {
  const runtime = useExternalStoreRuntime<ThreadMessage>({
    isRunning: false,
    messages,
    onNew: async () => {}
  })

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <Thread profile={profile} sessionKey={sessionKey} />
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

  // A background/keep-alive tile can be owned by a DIFFERENT profile than the
  // window's ambient active one (multi-profile Bot mode: several tabs, each
  // bound to its own bot/profile, mounted at once). Without `profile` threaded
  // through, every mounted instance read/wrote the ambient `$activeProfile`
  // bucket regardless of which profile it actually belonged to, so a
  // background tile's remembered position lived under the WRONG profile's key
  // (or polluted it) and could never be found again.
  describe('per-tile profile scoping (background tile owned by a non-ambient profile)', () => {
    it('restores from the tile’s OWN profile bucket, not the ambient active one', async () => {
      setActiveProfile('default')
      // A position saved for a DIFFERENT profile than the window's ambient one.
      saveThreadScrollPosition('x', { fromBottom: 800, kind: 'offset' }, threadScrollStorageKey('other'))

      const { container } = render(<ScrollHarness messages={sessionMessages('x')} profile="other" sessionKey="x" />)

      await settleScroll()

      expect(viewportEl(container).scrollTop).toBe(SCROLL_H - 800 - CLIENT_H)
    })

    it('does not leak a background tile’s remembered position into the ambient profile’s view of the same session key', async () => {
      setActiveProfile('default')
      saveThreadScrollPosition('x', { fromBottom: 800, kind: 'offset' }, threadScrollStorageKey('other'))

      // No `profile` prop: the primary view, which correctly follows the
      // ambient active profile ('default') — it must NOT see 'other's position.
      const { container } = render(<ScrollHarness messages={sessionMessages('x')} sessionKey="x" />)

      await settleScroll()

      expect(viewportEl(container).scrollTop).toBe(SCROLL_H - CLIENT_H)
    })

    it('persists a background tile’s position under its OWN profile, not the ambient one', async () => {
      setActiveProfile('default')

      const { container, rerender } = render(
        <ScrollHarness messages={sessionMessages('y')} profile="other" sessionKey="y" />
      )
      const vp = viewportEl(container)

      await settleScroll()

      // Scroll away from the bottom so the cleanup below has a real offset to
      // record (a live state at the bottom records nothing new here).
      act(() => {
        vp.scrollTop = SCROLL_H - CLIENT_H - 400
        vp.dispatchEvent(new Event('scroll'))
      })
      await settleScroll()

      // Switching sessionKey commits the outgoing instance's cleanup, which
      // records its live state under ITS OWN (profile-scoped) storage key.
      rerender(<ScrollHarness messages={sessionMessages('z')} profile="other" sessionKey="z" />)
      await settleScroll()

      expect(getThreadScrollPosition('y', threadScrollStorageKey('other'))).toEqual({ fromBottom: 400, kind: 'offset' })
      expect(getThreadScrollPosition('y', threadScrollStorageKey('default'))).toBeUndefined()
    })
  })
})
