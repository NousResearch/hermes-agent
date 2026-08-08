import { AssistantRuntimeProvider, type ThreadMessage, useExternalStoreRuntime } from '@assistant-ui/react'
import { render } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import { PaneVisibleContext } from '@/components/pane-shell/pane-visibility'

import { Thread } from '.'

/**
 * Keep-alive tabs (tree-group.tsx) stay MOUNTED while inactive — the pane
 * layer hides with `visibility: hidden`, preserving its layout box so scroll
 * positions survive a tab round-trip. The TRANSCRIPT must not ride along: a
 * hidden pane's frozen message rows are a second, stale copy of the
 * conversation (the "stale snapshot + live copy" duplication of #81772) that
 * can paint through the visible thread when a descendant overrides the
 * inherited visibility. Prove the renderer holds exactly ONE message list at
 * any time — the active pane's — while every pane's viewport shell stays
 * mounted.
 */

const createdAt = new Date('2026-05-01T00:00:00.000Z')

const MESSAGES: ThreadMessage[] = [
  {
    id: 'user-1',
    role: 'user',
    content: [{ type: 'text', text: 'hello from the user' }],
    attachments: [],
    createdAt,
    metadata: { custom: {} }
  } as ThreadMessage,
  {
    id: 'assistant-1',
    role: 'assistant',
    content: [{ type: 'text', text: 'stable assistant reply' }],
    status: { type: 'complete', reason: 'stop' },
    createdAt,
    metadata: {
      unstable_state: null,
      unstable_annotations: [],
      unstable_data: [],
      steps: [],
      custom: {}
    }
  } as ThreadMessage
]

const messageIds = (root: ParentNode): string[] =>
  [...root.querySelectorAll<HTMLElement>('[data-message-id]')].map(el => el.dataset.messageId ?? '')

const viewports = (root: ParentNode): number => root.querySelectorAll('[data-slot="aui_thread-viewport"]').length

class NoopResizeObserver {
  observe() {}

  unobserve() {}

  disconnect() {}
}

vi.stubGlobal('ResizeObserver', NoopResizeObserver)
vi.stubGlobal('requestAnimationFrame', (callback: FrameRequestCallback) =>
  window.setTimeout(() => callback(performance.now()), 0)
)
vi.stubGlobal('cancelAnimationFrame', (id: number) => window.clearTimeout(id))
vi.stubGlobal('CSS', { escape: (str: string) => str })

Element.prototype.scrollTo = function scrollTo() {}

Element.prototype.animate = function animate() {
  return {
    cancel: () => {},
    finished: Promise.resolve()
  } as unknown as Animation
}

function stubOffsetDimension(
  prop: 'offsetHeight' | 'offsetWidth',
  clientProp: 'clientHeight' | 'clientWidth',
  fallback: number
) {
  const previous = Object.getOwnPropertyDescriptor(HTMLElement.prototype, prop)

  Object.defineProperty(HTMLElement.prototype, prop, {
    configurable: true,
    get() {
      return previous?.get?.call(this) || (this as HTMLElement)[clientProp] || fallback
    }
  })
}

stubOffsetDimension('offsetWidth', 'clientWidth', 800)
stubOffsetDimension('offsetHeight', 'clientHeight', 600)

/** One keep-alive tab: the pane layer's visibility context + a mounted Thread. */
function Pane({ active, label }: { active: boolean; label: string }) {
  const runtime = useExternalStoreRuntime<ThreadMessage>({
    messages: MESSAGES,
    isRunning: false,
    onNew: async () => {}
  })

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <PaneVisibleContext.Provider value={active}>
        {/* The tree-group layer: only the ACTIVE tab is visible; inactive tabs
            stay mounted under `data-pane-hidden`. */}
        <div data-pane={label} data-pane-hidden={active ? undefined : ''}>
          <Thread sessionKey={label} />
        </div>
      </PaneVisibleContext.Provider>
    </AssistantRuntimeProvider>
  )
}

function Stack({ activeTab }: { activeTab: 'a' | 'b' }) {
  return (
    <div data-session-anchor="test">
      <Pane active={activeTab === 'a'} label="tab-a" />
      <Pane active={activeTab === 'b'} label="tab-b" />
    </div>
  )
}

const oneCopy = (container: HTMLElement) => expect(messageIds(container).sort()).toEqual(['assistant-1', 'user-1'])

describe('keep-alive message list', () => {
  it('renders the transcript in exactly one pane at a time', () => {
    const { container, rerender } = render(<Stack activeTab="a" />)

    // Tab A is active: its transcript renders; tab B is a hidden keep-alive
    // layer whose viewport shell stays mounted but carries no rows.
    oneCopy(container)
    expect(messageIds(container)).toHaveLength(2)
    expect(viewports(container)).toBe(2)

    // Switching the active tab must not double the transcript: the outgoing
    // pane's rows unmount, the incoming pane's mount — still one copy.
    rerender(<Stack activeTab="b" />)

    oneCopy(container)
    expect(viewports(container)).toBe(2)
  })

  it('hides the transcript of an inactive pane under data-pane-hidden', () => {
    const { container } = render(<Stack activeTab="a" />)

    const hiddenLayer = container.querySelector<HTMLElement>('[data-pane="tab-b"]')!
    const activeLayer = container.querySelector<HTMLElement>('[data-pane="tab-a"]')!

    expect(messageIds(hiddenLayer)).toHaveLength(0)
    expect(messageIds(activeLayer)).toHaveLength(2)
  })
})

/** A lone pane (no tab stack): the default visible context must keep the
 *  transcript rendered, exactly as before the keep-alive gate. */
function LoneHarness() {
  const runtime = useExternalStoreRuntime<ThreadMessage>({
    messages: MESSAGES,
    isRunning: false,
    onNew: async () => {}
  })

  return (
    <div data-session-anchor="lone">
      <AssistantRuntimeProvider runtime={runtime}>
        <Thread sessionKey="lone" />
      </AssistantRuntimeProvider>
    </div>
  )
}

describe('keep-alive message list outside a tab stack', () => {
  it('renders the transcript when no pane visibility context is present', () => {
    const { container } = render(<LoneHarness />)

    oneCopy(container)
  })
})
