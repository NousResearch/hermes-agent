// #42992: Settings → Appearance → Show full messages. Off (the default), sticky
// user bubbles clamp long prompts with a fade and expand on click. On, the body
// renders at natural height and a prompt taller than the pin budget drops the
// STICKY PIN (#39721: a pinned long prompt would eat the viewport its response
// needs, so it scrolls away as ordinary flow content instead).
//
// The measurement rides the app's shared ResizeObserver (see
// hooks/use-resize-observer.ts), so these tests drive a real RO delivery
// against the observed element (the bubble body wrapper), not a fake
// element: routing is by entry.target.
import { type AppendMessage, AssistantRuntimeProvider, ExportedMessageRepository, type ThreadMessage } from '@assistant-ui/react'
import { act, cleanup, render, waitFor } from '@testing-library/react'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import { useIncrementalExternalStoreRuntime } from '@/lib/incremental-external-store-runtime'
import { $showFullUserMessages } from '@/store/show-full-user-messages'

import { assistantMessage, stubThreadEnvironment, stubThreadViewportSize, userMessage } from '../test-utils'

import { Thread } from '.'

stubThreadEnvironment()

afterEach(() => {
  cleanup()
  $showFullUserMessages.set(false)
})

stubThreadViewportSize()

function Harness({ onEdit, text }: { onEdit: (message: AppendMessage) => Promise<void>; text: string }) {
  const repository = ExportedMessageRepository.fromArray([userMessage('user-1', text), assistantMessage()])

  const runtime = useIncrementalExternalStoreRuntime<ThreadMessage>({
    messageRepository: repository,
    isRunning: false,
    setMessages: () => {},
    onNew: async () => {},
    onEdit,
    onCancel: async () => {},
    onReload: async () => {}
  })

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <Thread cwd={null} gateway={null} sessionId="session-1" />
    </AssistantRuntimeProvider>
  )
}

// The app's shared ResizeObserver instance is created lazily from whatever
// global class is installed at first use, so capture every constructed
// instance: delivering an entry to all of them lets the shared observer's
// own routing (a WeakMap keyed by target element) pick the handlers that
// actually observe the target. Other instances (sticky-prompt-clip, Radix)
// just see a no-op resize.
const roInstances: Array<{ cb: ResizeObserverCallback }> = []

beforeAll(() => {
  vi.stubGlobal(
    'ResizeObserver',
    class {
      constructor(callback: ResizeObserverCallback) {
        roInstances.push({ cb: callback })
      }
      observe() {}
      unobserve() {}
      disconnect() {}
    }
  )
})

/** The bubble body wrapper the component refs for its natural-height RO. */
function bodyWrapper(): HTMLElement | null {
  return document.querySelector('[data-role="user"] [class="min-h-[1.25rem]"]')
}

/** Report the bubble body's natural height (px) through a real RO delivery. */
function deliverBodyHeight(height: number) {
  const body = bodyWrapper()

  expect(body).toBeTruthy()

  const entries = [
    {
      target: body,
      borderBoxSize: [{ blockSize: height, inlineSize: 400 }],
      contentRect: { height, width: 400 }
    }
  ] as unknown as ResizeObserverEntry[]

  for (const instance of roInstances) {
    instance.cb(entries, {} as ResizeObserver)
  }
}

/** The sticky root the transcript pins while a turn is read. */
function userRoot(): HTMLElement | null {
  return document.querySelector('[data-slot="aui_user-message-root"]')
}

/** The body renders as one whitespace-pre-line span: assert on the whole
 *  prompt text, not per-line elements. */
async function renderPrompt(text: string) {
  render(<Harness onEdit={vi.fn(async () => {})} text={text} />)

  await waitFor(() => {
    expect(userRoot()?.textContent).toContain(text.split('\n').at(-1))
  })
}

const LONG = Array.from({ length: 12 }, (_, i) => `line ${i + 1}`).join('\n')

describe('user bubble default (Show full messages off)', () => {
  it('clamps a long prompt and keeps it pinned', async () => {
    await renderPrompt(LONG)

    await act(async () => {
      deliverBodyHeight(240)
    })

    expect(document.querySelector('.sticky-human-clamp')).toBeTruthy()
    expect(document.querySelector('[data-clamped="true"]')).toBeTruthy()
    expect(userRoot()?.classList.contains('sticky')).toBe(true)
  })
})

describe('user bubble full render (#42992, Show full messages on)', () => {
  beforeEach(() => {
    $showFullUserMessages.set(true)
  })

  it('renders every line of a long prompt — no clamp, no fade', async () => {
    await renderPrompt(LONG)

    // jsdom resolves no line-height, so the component's fallback
    // (1.5 * font-size || 20) yields 20px/line: 12 lines = 240px, far past
    // the 4-line pin budget (81px).
    await act(async () => {
      deliverBodyHeight(240)
    })

    // The body is NOT wrapped in the clamp box: no max-height, no mask.
    expect(document.querySelector('.sticky-human-clamp')).toBeNull()
    expect(document.querySelector('[data-clamped]')).toBeNull()

    // The full text really is in the DOM.
    expect(userRoot()?.textContent).toContain('line 1')
    expect(userRoot()?.textContent).toContain('line 12')
  })

  it('drops the sticky pin for a prompt taller than the pin budget (#39721 stays safe)', async () => {
    await renderPrompt(LONG)

    // A pinned 240px prompt would eat the viewport its response needs, so a
    // too-tall prompt renders as ordinary flow content instead.
    await act(async () => {
      deliverBodyHeight(240)
    })

    expect(userRoot()?.classList.contains('sticky')).toBe(false)
  })

  it('keeps pinning a prompt inside the pin budget', async () => {
    await renderPrompt('short prompt')

    // 1 line = 20px, inside the 81px budget.
    await act(async () => {
      deliverBodyHeight(20)
    })

    expect(userRoot()?.classList.contains('sticky')).toBe(true)
  })
})
