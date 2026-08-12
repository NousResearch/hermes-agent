import {
  type AppendMessage,
  AssistantRuntimeProvider,
  ExportedMessageRepository,
  type ThreadMessage
} from '@assistant-ui/react'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { useIncrementalExternalStoreRuntime } from '@/lib/incremental-external-store-runtime'

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

function userMessage(): ThreadMessage {
  return {
    id: 'user-1',
    role: 'user',
    content: [{ type: 'text', text: 'edit me please' }],
    attachments: [],
    createdAt,
    metadata: { custom: {} }
  } as ThreadMessage
}

function assistantMessage(): ThreadMessage {
  return {
    id: 'assistant-1',
    role: 'assistant',
    content: [{ type: 'text', text: 'done' }],
    status: { type: 'complete', reason: 'stop' },
    createdAt,
    metadata: { unstable_state: null, unstable_annotations: [], unstable_data: [], steps: [], custom: {} }
  } as ThreadMessage
}

function Harness({ onEdit }: { onEdit: (message: AppendMessage) => Promise<void> }) {
  const repository = ExportedMessageRepository.fromArray([userMessage(), assistantMessage()])

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

// Regression for the desktop "editing a message, clicking the arrow does
// nothing — I have to click revert" report.
//
// On macOS a <button> does NOT take DOM focus on mousedown, so clicking the
// send arrow blurs the contenteditable (relatedTarget = null). The blur
// schedules an 80ms timer that cancels the edit, tearing down the assistant-ui
// edit-composer core. When the click's send() then runs, and again when the
// blur timer fires, cancel()/send() on a torn-down core throw "Composer is not
// available". The throw wedged the arrow (submitting stuck true) so only revert
// worked.
//
// The blur throw fires from a real setTimeout, which jsdom routes to Node as an
// `uncaughtException` (not a DOM error event), so a window 'error' probe misses
// it. Capture process-level uncaught exceptions for the duration of the gesture
// instead — an unguarded throw registers here and fails the test.
describe('edit send arrow — macOS click gesture (blur races cancel)', () => {
  it('sends without an uncaught "Composer is not available" when the arrow-click blurs the editor', async () => {
    const uncaught: unknown[] = []

    const onUncaught = (err: unknown) => {
      uncaught.push(err)
    }

    process.on('uncaughtException', onUncaught)

    try {
      const onEdit = vi.fn(async () => {})
      render(<Harness onEdit={onEdit} />)

      fireEvent.click(await screen.findByRole('button', { name: 'Edit message' }))
      const editor = await screen.findByRole('textbox', { name: 'Edit message' })

      await act(async () => {
        editor.focus()
        editor.textContent = 'edited then clicked the arrow'
        fireEvent.input(editor)
      })

      const send = await screen.findByRole('button', { name: 'Send edited message' })

      // The real gesture: mousedown on the arrow (no focus on macOS) blurs the
      // editor to <body>, then the click fires the send, then the blur's 80ms
      // cancel timer runs on the now-torn-down core. Wait past 80ms so the timer
      // completes within the captured window.
      await act(async () => {
        fireEvent.pointerDown(send)
        editor.blur()
        fireEvent.click(send)
        await new Promise(resolve => setTimeout(resolve, 200))
      })

      await waitFor(() => {
        expect(onEdit).toHaveBeenCalledTimes(1)
      })

      const composerErrors = uncaught.filter(err => /Composer is not available/.test(String(err)))

      expect(composerErrors).toEqual([])
    } finally {
      process.off('uncaughtException', onUncaught)
    }
  })
})
