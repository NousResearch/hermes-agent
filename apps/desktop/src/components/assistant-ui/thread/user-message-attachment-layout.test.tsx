// The attachment row on a user message must render as a flow sibling BELOW the
// sticky user bubble, with NO negative top margin — otherwise its top is pulled
// up under the bubble's opaque `sticky z-40` background and the chips/thumbnails
// get painted over (the clipping bug, #66260). This renders the real UserMessage
// through the app's assistant-ui runtime (same harness as user-message-edit) and
// asserts the DOM contract that locks the fix in: re-adding any `-mt-*` on the
// attachment row would fail here.
//
// Note: jsdom has no layout engine, so this asserts the structural/class
// contract (sibling ordering + absence of negative margin) rather than measured
// pixels. That is the exact thing the regression toggled, and it can't drift
// silently the way a screenshot can.
import { AssistantRuntimeProvider, type ThreadMessage, useExternalStoreRuntime } from '@assistant-ui/react'
import { cleanup, render, screen, waitFor } from '@testing-library/react'
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

afterEach(() => {
  cleanup()
})

function userMessageWithAttachment(): ThreadMessage {
  return {
    id: 'user-1',
    role: 'user',
    content: [{ type: 'text', text: 'here you go' }],
    attachments: [],
    createdAt,
    // The attachment row is driven by metadata.custom.attachmentRefs
    // (see messageAttachmentRefs in user-message.tsx).
    metadata: { custom: { attachmentRefs: ['@file:.hermes/desktop-attachments/文档.pdf'] } }
  } as ThreadMessage
}

function assistantMessage(): ThreadMessage {
  return {
    id: 'assistant-1',
    role: 'assistant',
    content: [{ type: 'text', text: 'done' }],
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
}

function Harness() {
  const runtime = useExternalStoreRuntime<ThreadMessage>({
    messages: [userMessageWithAttachment(), assistantMessage()],
    isRunning: false,
    onNew: async () => {},
    onEdit: async () => {}
  })

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <Thread />
    </AssistantRuntimeProvider>
  )
}

describe('user-message attachment row layout (#66260)', () => {
  it('renders the attachment row with no negative top margin so it is not clipped by the sticky bubble', async () => {
    const { container } = render(<Harness />)

    // The attachment ref renders through DirectiveContent; the row is its
    // wrapping div. Find it by the directive text the row contains.
    await waitFor(() => expect(screen.getByText(/文档\.pdf/)).toBeTruthy())
    const directive = screen.getByText(/文档\.pdf/)

    // Walk up to the attachment row div (the flex-wrap container).
    let row: HTMLElement | null = directive
    while (row && !(row.className && row.className.includes('flex-wrap'))) {
      row = row.parentElement
    }
    expect(row).not.toBeNull()

    // Contract 1: no negative top margin class of any size. This is the exact
    // thing the bug toggled (-mt-3 → clip; -mt-1 → partial clip). Any -mt-*
    // pulls the row under the opaque sticky layer.
    expect(row!.className).not.toMatch(/-mt-\d/)

    // Contract 2: the row still exists as its own spacing-bearing element
    // (mb-2 kept) — proving we removed the negative margin, not the row.
    expect(row!.className).toContain('mb-2')

    // Contract 3: the attachment row is a following sibling of the sticky
    // bubble container (role=user, sticky), i.e. it sits BELOW it in flow so
    // it scrolls away rather than riding along — the layout the fix relies on.
    const sticky = container.querySelector('[data-slot="aui_user-message-root"]')
    expect(sticky).not.toBeNull()
    expect(sticky!.className).toContain('sticky')
    // row must come after the sticky container in document order.
    const position = sticky!.compareDocumentPosition(row!)
    expect(position & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
    // and must NOT be a descendant of the sticky container (it's a flow sibling).
    expect(sticky!.contains(row!)).toBe(false)
  })
})
