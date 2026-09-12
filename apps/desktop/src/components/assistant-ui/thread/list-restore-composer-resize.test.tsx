import { AssistantRuntimeProvider, type ThreadMessage, useExternalStoreRuntime } from '@assistant-ui/react'
import { act, render } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { rescopeConnectionScopedStores } from '@/lib/connection-scoped'
import { setActiveProfile } from '@/store/profile'
import { saveThreadScrollPosition, threadScrollTargetTop } from '@/store/thread-scroll'

import { stubThreadEnvironment, stubThreadViewportSize } from '../test-utils'

import { Thread } from '.'

// Node 25 exposes a localStorage accessor that is neither undefined nor a
// Storage (no clear/setItem). Force an in-memory store so persist/restore works.
{
  const store = new Map<string, string>()
  const storage: Storage = {
    get length() {
      return store.size
    },
    clear: () => store.clear(),
    getItem: (key: string) => store.get(String(key)) ?? null,
    key: (index: number) => [...store.keys()][index] ?? null,
    removeItem: (key: string) => void store.delete(String(key)),
    setItem: (key: string, value: string) => void store.set(String(key), String(value))
  }

  for (const target of [globalThis, globalThis.window].filter(Boolean)) {
    Object.defineProperty(target, 'localStorage', { configurable: true, value: storage, writable: true })
  }
}

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
    const target = document.body
    this.callback(
      [
        {
          borderBoxSize: [],
          contentBoxSize: [],
          contentRect: { bottom: 1, height: 1, left: 0, right: 1, toJSON: () => ({}), top: 0, width: 1, x: 0, y: 0 } as DOMRect,
          devicePixelContentBoxSize: [],
          target
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
const FROM_BOTTOM = 800
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

function ScrollHarness({ messages, sessionKey }: { messages: ThreadMessage[]; sessionKey: string | null }) {
  const runtime = useExternalStoreRuntime<ThreadMessage>({
    isRunning: false,
    messages,
    onNew: async () => {}
  })

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <Thread clampToComposer sessionKey={sessionKey} />
    </AssistantRuntimeProvider>
  )
}

describe('list post-settle restore vs composer-only resize', () => {
  it('does not rewrite scrollTop when only composer clearance grows; content growth still restores', async () => {
    saveThreadScrollPosition('a', { fromBottom: FROM_BOTTOM, kind: 'offset' })

    const { container } = render(<ScrollHarness messages={sessionMessages('a')} sessionKey="a" />)
    const vp = viewportEl(container)

    await settleScroll()

    const restoredTop = SCROLL_H - CLIENT_H - FROM_BOTTOM
    expect(vp.scrollTop).toBe(restoredTop)

    const userRoot = container.querySelector('[data-slot="aui_user-message-root"]')
    expect(userRoot).toBeTruthy()
    expect(container.querySelector(`[data-slot="${CLEARANCE_SLOT}"]`)).toBeTruthy()

    // Composer-only: clearance + viewport box move together; transcript rows do not.
    viewportScrollHeight = SCROLL_H + 80
    viewportClientHeight = CLIENT_H - 80
    clearanceHeight = 200
    fireContentResizes()

    expect(vp.scrollTop).toBe(restoredTop)
    expect(container.querySelector('[data-slot="aui_user-message-root"]')).toBe(userRoot)

    // CONTROL: real transcript growth must still re-pin the frozen fromBottom.
    viewportScrollHeight = SCROLL_H + 80 + 600
    fireContentResizes()

    expect(vp.scrollTop).toBe(
      threadScrollTargetTop({ fromBottom: FROM_BOTTOM, kind: 'offset' }, { clientHeight: viewportClientHeight, scrollHeight: viewportScrollHeight })
    )
  })
})
