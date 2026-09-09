import { AssistantRuntimeProvider, type ThreadMessage, useExternalStoreRuntime } from '@assistant-ui/react'
import type { ReactNode } from 'react'
import { vi } from 'vitest'

import { stubResizeObserver } from '@/test/jsdom'

/** Fixed clock for message fixtures, so nothing sorts by "now". */
export const createdAt = new Date('2026-05-01T00:00:00.000Z')

/** The browser APIs the transcript renders against — a resize observer,
 *  animation frames, `CSS.escape` for its selector lookups, and the scroll and
 *  WAAPI calls the message list makes on mount. jsdom has none of them, and a
 *  thread that cannot mount fails every assertion in the file for the wrong
 *  reason. */
export function stubThreadEnvironment() {
  stubResizeObserver()

  vi.stubGlobal('requestAnimationFrame', (callback: FrameRequestCallback) =>
    window.setTimeout(() => callback(performance.now()), 0)
  )
  vi.stubGlobal('cancelAnimationFrame', (id: number) => window.clearTimeout(id))
  vi.stubGlobal('CSS', { escape: (value: string) => value })

  Element.prototype.scrollTo = function scrollTo() {}

  Element.prototype.animate = function animate() {
    return { cancel() {}, finished: Promise.resolve() } as unknown as Animation
  }
}

/** Give jsdom a viewport that the thread's virtualizer treats as scrollable.
 *
 *  jsdom reports every `offsetWidth`/`offsetHeight` as 0, which makes the
 *  message list measure itself as having no room and skip the rendering paths
 *  these tests are about. The stub falls through to a real value when one
 *  exists, so a test that sets its own dimensions still wins. */
export function stubThreadViewportSize() {
  const stub = (prop: 'offsetHeight' | 'offsetWidth', clientProp: 'clientHeight' | 'clientWidth', fallback: number) => {
    const previous = Object.getOwnPropertyDescriptor(HTMLElement.prototype, prop)

    Object.defineProperty(HTMLElement.prototype, prop, {
      configurable: true,
      get() {
        return previous?.get?.call(this) || (this as HTMLElement)[clientProp] || fallback
      }
    })
  }

  stub('offsetWidth', 'clientWidth', 800)
  stub('offsetHeight', 'clientHeight', 600)
}

/** Wrap a transcript subtree in a runtime driven by a fixed message list, the
 *  way the app's external store drives it: the thread is running while the tail
 *  message is. */
export function ThreadRuntime({ children, messages }: { children: ReactNode; messages: ThreadMessage[] }) {
  const runtime = useExternalStoreRuntime<ThreadMessage>({
    messages,
    isRunning: messages.at(-1)?.status?.type === 'running',
    onNew: async () => {}
  })

  return <AssistantRuntimeProvider runtime={runtime}>{children}</AssistantRuntimeProvider>
}

export function userMessage(id = 'user-1', text = 'edit me please'): ThreadMessage {
  return {
    id,
    role: 'user',
    content: [{ type: 'text', text }],
    attachments: [],
    createdAt,
    metadata: { custom: {} }
  } as ThreadMessage
}

export function assistantMessage(): ThreadMessage {
  return {
    id: 'assistant-1',
    role: 'assistant',
    content: [{ type: 'text', text: 'done' }],
    status: { type: 'complete', reason: 'stop' },
    createdAt,
    metadata: { unstable_state: null, unstable_annotations: [], unstable_data: [], steps: [], custom: {} }
  } as ThreadMessage
}
