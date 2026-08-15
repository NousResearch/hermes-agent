import { AssistantRuntimeProvider, type ThreadMessage, useExternalStoreRuntime } from '@assistant-ui/react'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $displayTimestamps } from '@/store/display-timestamps'
import { clearAllPrompts, setApprovalRequest } from '@/store/prompts'
import { $activeSessionId } from '@/store/session'
import { clearDismissedToolRows } from '@/store/tool-dismiss'
import { $toolDisclosureStates } from '@/store/tool-view'

import { Thread } from '../thread'
import { formatTimelineRange } from '../thread/timestamp'

// Timeline timestamps render only when `display.timestamps` is enabled.
$displayTimestamps.set(true)

// A run of tool calls collapses to a one-line summary once it has settled, but
// a run with anything still pending always renders its rows. That rule is what
// keeps the "approval must never be buried" bug fixed: an inline ApprovalBar
// only ever exists on a pending tool, and a pending tool's run is never behind
// a chevron. These cover both halves — the collapse itself, and the approval
// staying in the visual flow.

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
// pending terminal (no result), rendered as a flat two-row list.
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
  } as unknown as ThreadMessage
}

function pendingOnlyMessage(): ThreadMessage {
  return {
    id: 'assistant-pending-only',
    role: 'assistant',
    content: [
      {
        type: 'tool-call',
        toolCallId: 'term-only',
        toolName: 'terminal',
        args: { command: 'sleep 10' },
        argsText: JSON.stringify({ command: 'sleep 10' })
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
  } as unknown as ThreadMessage
}

function completedOnlyMessage(): ThreadMessage {
  return {
    id: 'assistant-completed-only',
    role: 'assistant',
    content: [
      {
        type: 'tool-call',
        toolCallId: 'read-only',
        toolName: 'read_file',
        args: { path: '/etc/hosts' },
        argsText: JSON.stringify({ path: '/etc/hosts' }),
        timestamp: createdAt.getTime() / 1000 + 10.125,
        completedAt: createdAt.getTime() / 1000 + 12.875,
        result: { content: '127.0.0.1 localhost' }
      }
    ],
    status: { type: 'complete', reason: 'stop' },
    createdAt,
    metadata: {
      unstable_state: null,
      unstable_annotations: [],
      unstable_data: [],
      steps: [],
      custom: {}
    }
  } as unknown as ThreadMessage
}

function failedOnlyMessage(): ThreadMessage {
  return {
    id: 'assistant-failed-only',
    role: 'assistant',
    content: [
      {
        type: 'tool-call',
        toolCallId: 'term-failed',
        toolName: 'terminal',
        args: { command: 'exit 1' },
        argsText: JSON.stringify({ command: 'exit 1' }),
        isError: true,
        result: { stderr: 'boom' }
      }
    ],
    status: { type: 'complete', reason: 'stop' },
    createdAt,
    metadata: {
      unstable_state: null,
      unstable_annotations: [],
      unstable_data: [],
      steps: [],
      custom: {}
    }
  } as unknown as ThreadMessage
}

// Two settled activity calls in a row, so the run earns a summary line and
// collapses behind it.
function settledRunMessage(): ThreadMessage {
  return {
    id: 'assistant-settled-run',
    role: 'assistant',
    content: [
      {
        type: 'tool-call',
        toolCallId: 'read-2',
        toolName: 'read_file',
        args: { path: '/repo/src/wiring.tsx' },
        argsText: JSON.stringify({ path: '/repo/src/wiring.tsx' }),
        timestamp: createdAt.getTime() / 1000 + 20,
        completedAt: createdAt.getTime() / 1000 + 21,
        result: { content: 'export const Wiring = () => null' }
      },
      {
        type: 'tool-call',
        toolCallId: 'term-3',
        toolName: 'terminal',
        args: { command: 'ls -la' },
        argsText: JSON.stringify({ command: 'ls -la' }),
        timestamp: createdAt.getTime() / 1000 + 22,
        completedAt: createdAt.getTime() / 1000 + 23.5,
        result: { exit_code: 0, stdout: 'wiring.tsx' }
      }
    ],
    status: { type: 'complete', reason: 'stop' },
    createdAt,
    metadata: {
      unstable_state: null,
      unstable_annotations: [],
      unstable_data: [],
      steps: [],
      custom: {}
    }
  } as unknown as ThreadMessage
}

// Activity, an edit, then more activity — all adjacent, so assistant-ui hands
// the whole stretch over as one group. The edit is the deliverable and has to
// survive that as its own card.
function editBetweenRunsMessage(): ThreadMessage {
  return {
    id: 'assistant-edit-between-runs',
    role: 'assistant',
    content: [
      {
        type: 'tool-call',
        toolCallId: 'read-5',
        toolName: 'read_file',
        args: { path: '/repo/src/a.ts' },
        argsText: JSON.stringify({ path: '/repo/src/a.ts' }),
        result: { content: 'a' }
      },
      {
        type: 'tool-call',
        toolCallId: 'search-3',
        toolName: 'search_files',
        args: { query: 'toolRuns' },
        argsText: JSON.stringify({ query: 'toolRuns' }),
        result: { hits: [] }
      },
      {
        type: 'tool-call',
        toolCallId: 'patch-2',
        toolName: 'patch',
        args: { path: '/repo/src/wiring.tsx' },
        argsText: JSON.stringify({ path: '/repo/src/wiring.tsx' }),
        result: { path: '/repo/src/wiring.tsx', inline_diff: '--- a\n+++ b\n+added line\n-removed line' }
      },
      {
        type: 'tool-call',
        toolCallId: 'read-6',
        toolName: 'read_file',
        args: { path: '/repo/src/b.ts' },
        argsText: JSON.stringify({ path: '/repo/src/b.ts' }),
        result: { content: 'b' }
      },
      {
        type: 'tool-call',
        toolCallId: 'term-4',
        toolName: 'terminal',
        args: { command: 'ls' },
        argsText: JSON.stringify({ command: 'ls' }),
        result: { exit_code: 0 }
      }
    ],
    status: { type: 'complete', reason: 'stop' },
    createdAt,
    metadata: {
      unstable_state: null,
      unstable_annotations: [],
      unstable_data: [],
      steps: [],
      custom: {}
    }
  } as unknown as ThreadMessage
}

// A finished turn that left a call without a result — interrupted, or the
// result landed elsewhere. The run is history and has to behave like it.
function abandonedRunMessage(): ThreadMessage {
  return {
    id: 'assistant-abandoned-run',
    role: 'assistant',
    content: [
      {
        type: 'tool-call',
        toolCallId: 'read-3',
        toolName: 'read_file',
        args: { path: '/repo/src/status.tsx' },
        argsText: JSON.stringify({ path: '/repo/src/status.tsx' }),
        result: { content: 'export const Status = () => null' }
      },
      {
        type: 'tool-call',
        toolCallId: 'search-1',
        toolName: 'search_files',
        args: { query: 'toolRuns' },
        argsText: JSON.stringify({ query: 'toolRuns' })
      }
    ],
    status: { type: 'complete', reason: 'stop' },
    createdAt,
    metadata: {
      unstable_state: null,
      unstable_annotations: [],
      unstable_data: [],
      steps: [],
      custom: {}
    }
  } as unknown as ThreadMessage
}

// The gap between one sequential call finishing and the next arriving: the
// turn is still running, the run is still the tail, but for this instant every
// call has a result.
function betweenSequentialCallsMessage(): ThreadMessage {
  return {
    id: 'assistant-between-calls',
    role: 'assistant',
    content: [
      {
        type: 'tool-call',
        toolCallId: 'term-a',
        toolName: 'terminal',
        args: { command: 'sleep 2; echo alpha' },
        argsText: JSON.stringify({ command: 'sleep 2; echo alpha' }),
        result: { exit_code: 0, stdout: 'alpha' }
      },
      {
        type: 'tool-call',
        toolCallId: 'term-b',
        toolName: 'terminal',
        args: { command: 'sleep 2; echo bravo' },
        argsText: JSON.stringify({ command: 'sleep 2; echo bravo' }),
        result: { exit_code: 0, stdout: 'bravo' }
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
  } as unknown as ThreadMessage
}

// Still streaming, but the agent has moved past its first run and left both of
// its calls unresolved. Only the run at the tail is still live.
function movedOnMessage(): ThreadMessage {
  return {
    id: 'assistant-moved-on',
    role: 'assistant',
    content: [
      {
        type: 'tool-call',
        toolCallId: 'read-4',
        toolName: 'read_file',
        args: { path: '/repo/src/status.tsx' },
        argsText: JSON.stringify({ path: '/repo/src/status.tsx' })
      },
      {
        type: 'tool-call',
        toolCallId: 'search-2',
        toolName: 'search_files',
        args: { query: 'toolRuns' },
        argsText: JSON.stringify({ query: 'toolRuns' })
      },
      { type: 'text', text: 'Let me read the rest of the file.' },
      {
        type: 'tool-call',
        toolCallId: 'term-2',
        toolName: 'terminal',
        args: { command: 'ls' },
        argsText: JSON.stringify({ command: 'ls' })
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
  } as unknown as ThreadMessage
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
  clearAllPrompts()
  $activeSessionId.set('sess-1')
  $toolDisclosureStates.set({})
  clearDismissedToolRows()
})

afterEach(() => {
  cleanup()
  clearAllPrompts()
  $activeSessionId.set(null)
  clearDismissedToolRows()
})

describe('settled tool run', () => {
  it('collapses to a summary line naming the work', async () => {
    const { container } = render(<GroupHarness message={settledRunMessage()} />)

    expect(await screen.findByText('Explored wiring.tsx, ran 1 command')).toBeTruthy()
    expect(container.querySelectorAll('[data-tool-row]')).toHaveLength(0)
  })

  it('expands to the underlying rows when the summary is clicked', async () => {
    const { container } = render(<GroupHarness message={settledRunMessage()} />)

    fireEvent.click(await screen.findByText('Explored wiring.tsx, ran 1 command'))

    await waitFor(() => {
      expect(container.querySelectorAll('[data-tool-row]').length).toBeGreaterThan(0)
    })
  })

  it('leaves a lone tool call as its own row, with no summary above it', async () => {
    const { container } = render(<GroupHarness message={completedOnlyMessage()} />)

    await waitFor(() => {
      expect(container.querySelectorAll('[data-tool-row]').length).toBe(1)
    })
    expect(container.querySelector('[data-tool-summary]')).toBeNull()
  })
})

// A diff is what the user reviews, so it is never what gets summarized away.
// It stays on screen at the point in the turn where it happened, with the
// activity either side of it collapsing around it.
describe('a file edit among ordinary activity', () => {
  it('stays visible between the two runs it interrupted', async () => {
    const { container } = render(<GroupHarness message={editBetweenRunsMessage()} />)

    await screen.findByText('Explored 2 files')

    const shape = [...container.querySelectorAll('[data-tool-summary],[data-tool-row]')].map(node =>
      node.hasAttribute('data-tool-summary') ? 'summary' : 'row'
    )

    expect(shape).toEqual(['summary', 'row', 'summary'])
  })

  it('keeps the diff itself on screen rather than behind the summary', async () => {
    const { container } = render(<GroupHarness message={editBetweenRunsMessage()} />)

    await waitFor(() => {
      expect(container.querySelector('[data-tool-row][data-file-edit]')).not.toBeNull()
    })
  })
})

// The transcript rests its scaffolding at a fade, keyed off one attribute. A
// surface that renders without it is brighter than everything around it, which
// is how two adjacent, identical rows came to sit at two opacities.
describe('transcript fade', () => {
  it('marks every row and summary as scaffolding', async () => {
    const { container } = render(<GroupHarness message={editBetweenRunsMessage()} />)

    await screen.findByText('Explored 2 files')

    const unmarked = [...container.querySelectorAll('[data-tool-summary],[data-tool-row]')].filter(
      node => !node.hasAttribute('data-conversation-scaffold')
    )

    expect(unmarked).toHaveLength(0)
  })
})

describe('live tool run', () => {
  it('keeps its rows on screen instead of hiding them behind the summary', async () => {
    const { container } = render(<GroupHarness message={groupedPendingMessage()} />)

    await waitFor(() => {
      expect(container.querySelectorAll('[data-tool-row]').length).toBeGreaterThan(0)
    })
  })

  it('cannot be collapsed while a tool is still running', async () => {
    const { container } = render(<GroupHarness message={groupedPendingMessage()} />)

    await waitFor(() => {
      expect(container.querySelector('[data-tool-summary]')).not.toBeNull()
    })

    expect(container.querySelector('[data-tool-summary] button[aria-expanded]')).toBeNull()
  })

  // Liveness used to also require an unresolved call, which is false for the
  // instant between one sequential call finishing and the next arriving — so a
  // string of commands settled and re-opened between every one, unmounting the
  // ticker and dropping its reel back to the first row instead of scrolling.
  it('stays live in the gap between two sequential calls', async () => {
    const { container } = render(<GroupHarness message={betweenSequentialCallsMessage()} />)

    expect(await screen.findByText('Running 2 commands')).toBeTruthy()
    expect(container.querySelector('[data-tool-ticker]')).not.toBeNull()
    expect(container.querySelector('[data-tool-summary] button[aria-expanded]')).toBeNull()
  })

  // The ticker is a one-line window, so a row opened inside it had its output
  // sliced to that line and then ticked away by the next call. Opening a row
  // is a request to read it: the run gives up the window until it settles.
  it('drops the one-line window when a row inside it is opened', async () => {
    const { container } = render(<GroupHarness message={betweenSequentialCallsMessage()} />)

    await screen.findByText('Running 2 commands')

    const row = container.querySelector('[data-tool-ticker] [data-tool-row] button[aria-expanded="false"]')

    expect(row).not.toBeNull()

    fireEvent.click(row as Element)

    await waitFor(() => {
      expect(container.querySelector('[data-tool-ticker]')).toBeNull()
    })

    // ...and the row it opened is still on screen to be read.
    expect(container.querySelector('[data-tool-row][data-tool-open]')).not.toBeNull()
  })
})

// A run whose calls never resolved used to read as live forever, which stranded
// it in the present tense and — because a live run withholds its toggle — left
// it permanently expanded with no way to collapse it.
describe('tool run left unresolved', () => {
  it('settles with the turn rather than narrating work that stopped', async () => {
    const { container } = render(<GroupHarness message={abandonedRunMessage()} />)

    expect(await screen.findByText('Explored 2 files')).toBeTruthy()
    expect(container.querySelectorAll('[data-tool-row]')).toHaveLength(0)
    expect(container.querySelector('[data-tool-summary] button[aria-expanded]')).not.toBeNull()
  })

  it('settles once the agent moves on, even mid-turn', async () => {
    const { container } = render(<GroupHarness message={movedOnMessage()} />)

    expect(await screen.findByText('Explored 2 files')).toBeTruthy()
    expect(container.querySelector('[data-tool-summary] button[aria-expanded]')).not.toBeNull()
  })
})

describe('flat tool list approval surfacing', () => {
  it('renders no inline approval bar when there is no live approval', async () => {
    const { container } = render(<GroupHarness message={groupedPendingMessage()} />)

    // The pending terminal row mounts immediately, but its inline ApprovalBar
    // returns null while $approvalRequest is empty.
    await waitFor(() => {
      expect(container.querySelectorAll('[data-slot="tool-block"]').length).toBeGreaterThan(0)
    })
    expect(container.querySelector('[data-slot="tool-approval-inline"]')).toBeNull()
  })

  it('surfaces the approval inline and never under a hidden ancestor', async () => {
    setApprovalRequest({ command: 'rm -rf /tmp/x', description: 'dangerous command', sessionId: 'sess-1' })

    const { container } = render(<GroupHarness message={groupedPendingMessage()} />)

    await waitFor(() => {
      const bar = container.querySelector('[data-slot="tool-approval-inline"]')
      expect(bar).not.toBeNull()
      // Flat rows live directly in the flow — nothing should ever wrap the bar
      // in a `hidden` subtree.
      expect(bar?.closest('[hidden]')).toBeNull()
    })
  })

  it('lets completed tool rows be dismissed', async () => {
    const { container } = render(<GroupHarness message={completedOnlyMessage()} />)

    const dismiss = await screen.findByLabelText('Dismiss')

    expect(container.querySelectorAll('[data-slot="tool-block"]').length).toBeGreaterThan(0)

    fireEvent.click(dismiss)

    await waitFor(() => {
      expect(screen.queryByLabelText('Dismiss')).toBeNull()
    })
  })

  it('keeps a dismissed row hidden after a remount (virtualization)', async () => {
    // The thread virtualizes, so a row's component unmounts/remounts as it
    // scrolls. Dismissal must persist across that — component-local state would
    // forget it and the row would pop back. Simulate the remount by unmounting
    // and rendering the same message fresh.
    const first = render(<GroupHarness message={completedOnlyMessage()} />)

    fireEvent.click(await screen.findByLabelText('Dismiss'))

    await waitFor(() => {
      expect(screen.queryByLabelText('Dismiss')).toBeNull()
    })

    first.unmount()

    render(<GroupHarness message={completedOnlyMessage()} />)

    // The row is the only thing this message renders, so staying dismissed
    // means nothing comes back — including its dismiss control.
    await waitFor(() => {
      expect(screen.queryByLabelText('Dismiss')).toBeNull()
    })
  })

  it('lets failed tool rows be dismissed', async () => {
    render(<GroupHarness message={failedOnlyMessage()} />)

    const dismiss = await screen.findByLabelText('Dismiss')

    fireEvent.click(dismiss)

    await waitFor(() => {
      expect(screen.queryByLabelText('Dismiss')).toBeNull()
    })
  })

  it('does not show dismiss for pending tool rows', async () => {
    const { container } = render(<GroupHarness message={pendingOnlyMessage()} />)

    await waitFor(() => {
      expect(container.querySelectorAll('[data-slot="tool-block"]').length).toBeGreaterThan(0)
    })

    expect(screen.queryByLabelText('Dismiss')).toBeNull()
  })
})

describe('tool lifecycle timestamps', () => {
  it('shows the precise call and completion times on a settled tool row', async () => {
    const { container } = render(<GroupHarness message={completedOnlyMessage()} />)

    await screen.findByText(/Read/)

    const timestamps = Array.from(container.querySelectorAll('[data-slot="timeline-timestamp"]')).map(node =>
      node.textContent?.trim()
    )

    const startedAt = createdAt.getTime() / 1000 + 10.125

    expect(timestamps).toContain(formatTimelineRange(startedAt, createdAt.getTime() / 1000 + 12.875))
  })

  it('shows the full lifecycle range when settled calls are collapsed', async () => {
    const { container } = render(<GroupHarness message={settledRunMessage()} />)

    await waitFor(() => expect(container.querySelector('[data-tool-summary]')).toBeTruthy())

    const timestamps = Array.from(container.querySelectorAll('[data-slot="timeline-timestamp"]')).map(node =>
      node.textContent?.trim()
    )

    expect(timestamps).toContain(
      formatTimelineRange(createdAt.getTime() / 1000 + 20, createdAt.getTime() / 1000 + 23.5)
    )
  })
})
