// Click-to-expand link-preview chips (D7) mount as the AssistantLinkPreviews
// leaf next to AssistantPreviewEmbeds. Three contracts are pinned here:
//
//  1. Chips render ONLY once the turn settles — the '' while running branch
//     keeps the selector stable so per-token flushes skip the URL scan.
//  2. Chips render ONLY when the desktop bridge is present: without
//     `fetchLinkPreview` there is no fetch path, so the row is not rendered
//     at all (the collapsed chip would be a promise the app cannot keep).
//  3. A click on the chip goes through the bridge exactly once. No network
//     call happens for a mentioned-but-never-clicked URL (the card test
//     covers the expansion legs; here it is the message-level mount).
import { AssistantRuntimeProvider, type ThreadMessage, useExternalStoreRuntime } from '@assistant-ui/react'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { Thread } from '.'

const createdAt = new Date('2026-05-01T00:00:00.000Z')

class TestResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
}

vi.stubGlobal('ResizeObserver', TestResizeObserver)
vi.stubGlobal('requestAnimationFrame', (callback: FrameRequestCallback) =>
  window.setTimeout(() => callback(performance.now()), 0)
)
vi.stubGlobal('cancelAnimationFrame', (id: number) => window.clearTimeout(id))
vi.stubGlobal('CSS', { escape: (str: string) => str })

Element.prototype.scrollTo = function scrollTo() {}

Element.prototype.animate = function animate() {
  return { cancel() {}, finished: Promise.resolve() } as unknown as Animation
}

afterEach(() => {
  cleanup()
   
  ;(window as any).hermesDesktop = undefined
})

const assistantMetadata = { unstable_state: null, unstable_annotations: [], unstable_data: [], steps: [], custom: {} }

function user(id: string, text: string): ThreadMessage {
  return {
    id,
    role: 'user',
    content: [{ type: 'text', text }],
    attachments: [],
    createdAt,
    metadata: { custom: {} }
  } as ThreadMessage
}

function assistant(id: string, text: string, running: boolean): ThreadMessage {
  return {
    id,
    role: 'assistant',
    content: text ? [{ type: 'text', text }] : [],
    status: running ? { type: 'running' } : { type: 'complete', reason: 'stop' },
    createdAt,
    metadata: assistantMetadata
  } as ThreadMessage
}

function Harness({ messages }: { messages: ThreadMessage[] }) {
  const runtime = useExternalStoreRuntime<ThreadMessage>({
    messages,
    isRunning: messages.at(-1)?.status?.type === 'running',
    onNew: async () => {}
  })

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <Thread />
    </AssistantRuntimeProvider>
  )
}

const URL = 'https://example.com/release-notes'

function installBridge(fetchLinkPreview: unknown) {
  ;(window as unknown as { hermesDesktop: unknown }).hermesDesktop = {
    fetchLinkTitle: vi.fn().mockResolvedValue(''),
    openExternal: vi.fn().mockResolvedValue(undefined),
    fetchLinkPreview
  }
}

describe('assistant link-preview chips (D7)', () => {
  it('renders a chip for an external URL once the turn settles, and fetches only on click', async () => {
    const bridge = vi.fn().mockResolvedValue({
      ok: true,
      meta: { url: URL, title: 'Release Notes', description: '', imageUrl: '', fetchedAt: 1_000 }
    })

    installBridge(bridge)

    const { container } = render(
      <Harness messages={[user('u1', 'what shipped?'), assistant('a1', `Full notes at ${URL}`, false)]} />
    )

    const chip = await waitFor(() => {
      const found = container.querySelector('[data-link-preview="chip"]')

      expect(found).toBeTruthy()

      return found as HTMLElement
    })

    expect(bridge).not.toHaveBeenCalled()

    fireEvent.click(chip)

    await waitFor(() => expect(bridge).toHaveBeenCalledTimes(1))
    await waitFor(() => expect(bridge).toHaveBeenCalledWith(URL))
    expect(container.querySelector('[data-link-preview="card"]')).toBeTruthy()
  })

  it('does not render the chip while the turn is still running', async () => {
    installBridge(vi.fn())

    const { container } = render(
      <Harness messages={[user('u1', 'what shipped?'), assistant('a1', `Full notes at ${URL}`, true)]} />
    )

    await screen.findByText('Full notes at', { exact: false })

    expect(container.querySelector('[data-link-preview="chip"]')).toBeNull()
  })

  it('renders nothing without the desktop bridge, even when settled', async () => {
    ;(window as unknown as { hermesDesktop: unknown }).hermesDesktop = undefined

    const { container } = render(
      <Harness messages={[user('u1', 'what shipped?'), assistant('a1', `Full notes at ${URL}`, false)]} />
    )

    await screen.findByText('Full notes at', { exact: false })

    expect(container.querySelector('[data-link-preview="chip"]')).toBeNull()
    expect(container.querySelector('[data-link-preview="card"]')).toBeNull()
  })

  it('local URLs never produce a chip', async () => {
    installBridge(vi.fn())

    const { container } = render(
      <Harness
        messages={[user('u1', 'dev urls'), assistant('a1', 'dashboard at http://localhost:3000 and https://127.0.0.1:8080', false)]}
      />
    )

    await screen.findByText('dashboard at', { exact: false })

    expect(container.querySelector('[data-link-preview="chip"]')).toBeNull()
  })
})
