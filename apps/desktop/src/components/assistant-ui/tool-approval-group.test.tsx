import { AssistantRuntimeProvider, type ThreadMessage, useExternalStoreRuntime } from '@assistant-ui/react'
import { cleanup, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $approvalRequest } from '@/store/prompts'
import { $toolDisclosureStates } from '@/store/tool-view'

import { Thread } from './thread'

// Regression coverage for the "approval buried behind a collapsed tool group"
// bug. When 2+ tools group into a collapsed "Tool actions · N steps" row, the
// pending tool's inline ApprovalBar lives inside the group body — which is
// `hidden` until expanded. A live approval must surface WITHOUT the user
// expanding anything, so ToolGroupSlot force-opens its body while an approval
// targeting one of its pending tools is in flight.

const createdAt = new Date('2026-06-03T00:00:00.000Z')

const resizeObservers = new Set<TestResizeObserver>()

class TestResizeObserver {
  private target: Element | null = null

  constructor(private readonly callback: ResizeObserverCallback) {
    resizeObservers.add(this)
  }

  observe(target: Element) {
    this.target = target
  }

  unobserve() {}

  disconnect() {
    resizeObservers.delete(this)
  }
}

vi.stubGlobal('ResizeObserver', TestResizeObserver)
vi.stubGlobal('requestAnimationFrame', (callback: FrameRequestCallback) =>
  window.setTimeout(() => callback(performance.now()), 0)
)
vi.stubGlobal('cancelAnimationFrame', (id: number) => window.clearTimeout(id))

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

// A running assistant message with two tools: a completed read_file plus a
// pending terminal (no result). Two visible tools → ToolGroupSlot groups them
// behind a collapsed "Tool actions · 2 steps" header.
function groupedPendingMessage(): ThreadMessage {
  return {
    id: 'assistant-group-1',
    role: 'assistant',
    content: [
      {
        type: 'tool-call',
        toolCallId: 'read-1',
        toolName: 'read_file',
        args: { path: '/etc/hosts' },
        argsText: JSON.stringify({ path: '/etc/hosts' }),
        result: { content: '127.0.0.1 localhost' }
      },
      {
        type: 'tool-call',
        toolCallId: 'term-1',
        toolName: 'terminal',
        args: { command: 'rm -rf /tmp/x' },
        argsText: JSON.stringify({ command: 'rm -rf /tmp/x' })
      }
    ],
    status: { type: 'running' },
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

function GroupHarness({ message }: { message: ThreadMessage }) {
  const runtime = useExternalStoreRuntime<ThreadMessage>({
    messages: [message],
    isRunning: message.status?.type === 'running',
    onNew: async () => {}
  })

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <Thread />
    </AssistantRuntimeProvider>
  )
}

beforeEach(() => {
  $approvalRequest.set(null)
  $toolDisclosureStates.set({})
})

afterEach(() => {
  cleanup()
  $approvalRequest.set(null)
})

describe('ToolGroupSlot approval surfacing', () => {
  it('hides the grouped pending tool body when there is no approval', async () => {
    const { container } = render(<GroupHarness message={groupedPendingMessage()} />)

    // Group header renders collapsed; the inline approval strip lives in the
    // hidden body, so with no live approval it must not render at all (the
    // ApprovalBar returns null when $approvalRequest is empty).
    await waitFor(() => {
      expect(screen.getByText(/Tool actions/)).toBeTruthy()
    })
    expect(container.querySelector('[data-slot="tool-approval-inline"]')).toBeNull()
  })

  it('force-opens the group body so the approval surfaces without expanding', async () => {
    $approvalRequest.set({ command: 'rm -rf /tmp/x', description: 'dangerous command', sessionId: 'sess-1' })

    const { container } = render(<GroupHarness message={groupedPendingMessage()} />)

    // Even though the group defaults collapsed, the live approval forces the
    // body open so the inline controls are visible (and reachable, not in a
    // hidden subtree) immediately.
    await waitFor(() => {
      const bar = container.querySelector('[data-slot="tool-approval-inline"]')
      expect(bar).not.toBeNull()
      // The forced-open group body must not be hidden — assert no ancestor
      // carries the `hidden` attribute that would keep the bar off-screen.
      expect(bar?.closest('[hidden]')).toBeNull()
    })
  })
})
