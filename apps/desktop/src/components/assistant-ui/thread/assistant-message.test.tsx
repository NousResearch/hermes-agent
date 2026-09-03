// Bug #2: the Branch-in-new-chat button used to render unconditionally even
// when its handler was a no-op (session-tile.tsx passed `() => undefined`
// for branched/tiled chats, where nested branching isn't supported). That
// left a visibly clickable button that silently did nothing. The fix makes
// AssistantMessage's action bar hide the button entirely when no handler is
// supplied, matching how onDismissError/onRestoreToMessage already behave.
import { AssistantRuntimeProvider, type ThreadMessage, useExternalStoreRuntime } from '@assistant-ui/react'
import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { $displayTimestamps } from '@/store/display-timestamps'

import { stubThreadEnvironment } from '../test-utils'

import { formatClockTimestamp, formatTimelineRange } from './timestamp'

import { Thread } from '.'

// Timeline timestamps render only when `display.timestamps` is enabled.
$displayTimestamps.set(true)

const createdAt = new Date('2026-05-01T00:00:00.000Z')
const completedAt = createdAt.getTime() / 1000 + 1.25
stubThreadEnvironment()

afterEach(() => {
  cleanup()
})

function userMessage(): ThreadMessage {
  return {
    id: 'user-1',
    role: 'user',
    content: [{ type: 'text', text: 'question one' }],
    attachments: [],
    createdAt,
    metadata: { custom: { timelineTimestamp: createdAt.getTime() / 1000 } }
  } as unknown as ThreadMessage
}

function assistantMessage(): ThreadMessage {
  return {
    id: 'assistant-1',
    role: 'assistant',
    content: [
      {
        type: 'reasoning',
        text: 'checked carefully',
        timestamp: createdAt.getTime() / 1000 + 0.05,
        completedAt: createdAt.getTime() / 1000 + 0.1
      },
      {
        type: 'text',
        text: 'done',
        timestamp: createdAt.getTime() / 1000 + 0.125,
        completedAt: createdAt.getTime() / 1000 + 0.5
      }
    ],
    status: { type: 'complete', reason: 'stop' },
    createdAt,
    metadata: {
      unstable_state: null,
      unstable_annotations: [],
      unstable_data: [],
      steps: [],
      custom: { timelineCompletedAt: completedAt, timelineTimestamp: createdAt.getTime() / 1000 }
    }
  } as unknown as ThreadMessage
}

function Harness({
  assistant = assistantMessage(),
  onBranchInNewChat,
  user = userMessage()
}: {
  assistant?: ThreadMessage
  onBranchInNewChat?: (messageId: string) => void
  user?: ThreadMessage
}) {
  const runtime = useExternalStoreRuntime<ThreadMessage>({
    messages: [user, assistant],
    isRunning: false,
    onNew: async () => {}
  })

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <Thread onBranchInNewChat={onBranchInNewChat} />
    </AssistantRuntimeProvider>
  )
}

describe('AssistantMessage branch button visibility (bug #2 fix)', () => {
  it('shows the Branch in new chat button when a handler is provided (open chat)', async () => {
    render(<Harness onBranchInNewChat={() => undefined} />)

    expect(await screen.findByRole('button', { name: 'Branch in new chat' })).toBeTruthy()
  })

  it('hides the Branch in new chat button when no handler is provided (session-tile / branched chat)', async () => {
    render(<Harness />)

    // Wait for the assistant message to actually mount before asserting
    // absence, so a missing button isn't just a false negative from an
    // unrendered message.
    await screen.findByText('done')

    expect(screen.queryByRole('button', { name: 'Branch in new chat' })).toBeNull()
  })
})

describe('message timeline timestamps', () => {
  it('shows chat bubbles as a minute-precision clock and activity parts as precise ranges', async () => {
    const { container } = render(<Harness />)

    await screen.findByText('done')

    const stamps = Array.from(container.querySelectorAll('[data-slot="timeline-timestamp"]')).map(node =>
      node.textContent?.trim()
    )

    const startedAt = createdAt.getTime() / 1000

    // Bubble rows: "when was it sent" / "when did it land", no seconds.
    expect(stamps).toContain(formatClockTimestamp(startedAt))
    expect(stamps).toContain(formatClockTimestamp(completedAt))
    expect(stamps).not.toContain(formatTimelineRange(startedAt, completedAt))
    // The text part carries the prose, so it is a bubble row too — this is
    // the exact row that used to print `5:06:59.615 AM → 5:08:37.037 AM`.
    expect(stamps).not.toContain(formatTimelineRange(startedAt + 0.125, startedAt + 0.5))

    // Reasoning stays an activity boundary and keeps full precision.
    expect(stamps).toContain(formatTimelineRange(startedAt + 0.05, startedAt + 0.1))
  })

  it('renders the assistant landing clock from the completion time, not the send time', async () => {
    const startedAt = createdAt.getTime() / 1000
    // Deliberately crosses a minute boundary so "sent" and "landed" differ.
    const landedAt = startedAt + 130

    const assistant = {
      ...assistantMessage(),
      metadata: {
        ...assistantMessage().metadata,
        custom: { timelineCompletedAt: landedAt, timelineTimestamp: startedAt }
      }
    } as unknown as ThreadMessage

    const { container } = render(<Harness assistant={assistant} />)

    await screen.findByText('done')

    const stamps = Array.from(container.querySelectorAll('[data-slot="timeline-timestamp"]')).map(node =>
      node.textContent?.trim()
    )

    expect(formatClockTimestamp(landedAt)).not.toBe(formatClockTimestamp(startedAt))
    expect(stamps).toContain(formatClockTimestamp(landedAt))
  })

  it('shows one assistant clock when its prose part starts after the message', async () => {
    const startedAt = createdAt.getTime() / 1000
    // Live prose starts on the first text delta, after the message lifecycle.
    const partStartedAt = startedAt + 0.125
    const landedAt = startedAt + 130

    const assistant = {
      ...assistantMessage(),
      content: [{ completedAt: landedAt, text: 'done', timestamp: partStartedAt, type: 'text' }],
      metadata: {
        ...assistantMessage().metadata,
        custom: { timelineCompletedAt: landedAt, timelineTimestamp: startedAt }
      }
    } as unknown as ThreadMessage

    const { container } = render(<Harness assistant={assistant} />)

    await screen.findByText('done')

    const stamps = Array.from(container.querySelectorAll('[data-slot="timeline-timestamp"]')).map(node =>
      node.textContent?.trim()
    )

    expect(formatClockTimestamp(landedAt)).not.toBe(formatClockTimestamp(startedAt))
    expect(stamps.filter(stamp => stamp === formatClockTimestamp(landedAt))).toHaveLength(1)
    expect(stamps.filter(stamp => stamp === formatClockTimestamp(startedAt))).toHaveLength(1)
    expect(stamps).not.toContain(formatTimelineRange(startedAt, landedAt))
  })

  it('shows one assistant clock when reasoning precedes prose', async () => {
    const startedAt = createdAt.getTime() / 1000
    const landedAt = startedAt + 130

    const assistant = {
      ...assistantMessage(),
      content: [
        { completedAt: startedAt + 0.1, text: 'thinking', timestamp: startedAt + 0.05, type: 'reasoning' },
        { completedAt: landedAt, text: 'done', timestamp: startedAt + 0.125, type: 'text' }
      ],
      metadata: {
        ...assistantMessage().metadata,
        custom: { timelineCompletedAt: landedAt, timelineTimestamp: startedAt }
      }
    } as unknown as ThreadMessage

    const { container } = render(<Harness assistant={assistant} />)
    await screen.findByText('done')

    const stamps = Array.from(container.querySelectorAll('[data-slot="timeline-timestamp"]')).map(node =>
      node.textContent?.trim()
    )

    expect(stamps.filter(stamp => stamp === formatClockTimestamp(landedAt))).toHaveLength(1)
    expect(stamps).toContain(formatTimelineRange(startedAt + 0.05, startedAt + 0.1))
  })

  it('shows one assistant clock across multiple prose parts', async () => {
    const startedAt = createdAt.getTime() / 1000
    const landedAt = startedAt + 130

    const assistant = {
      ...assistantMessage(),
      content: [
        { completedAt: startedAt + 0.2, text: 'first', timestamp: startedAt + 0.1, type: 'text' },
        { completedAt: landedAt, text: 'second', timestamp: startedAt + 1, type: 'text' }
      ],
      metadata: {
        ...assistantMessage().metadata,
        custom: { timelineCompletedAt: landedAt, timelineTimestamp: startedAt }
      }
    } as unknown as ThreadMessage

    const { container } = render(<Harness assistant={assistant} />)
    await screen.findByText('second')

    const stamps = Array.from(container.querySelectorAll('[data-slot="timeline-timestamp"]')).map(node =>
      node.textContent?.trim()
    )

    expect(stamps.filter(stamp => stamp === formatClockTimestamp(landedAt))).toHaveLength(1)
  })

  it('shows one clock on each outbound agent delivery notice', async () => {
    const startedAt = createdAt.getTime() / 1000
    const landedAt = startedAt + 130

    const assistant = {
      ...assistantMessage(),
      content: [
        {
          args: {
            command: 'hermes -p badr chat --in ~ -c "Bot Chat" -Q -q "Message from 🤖 Abu Saud: review"'
          },
          argsText: '{}',
          completedAt: landedAt,
          result: { exit_code: 0, output: 'done' },
          timestamp: startedAt,
          toolCallId: 'delivery-1',
          toolName: 'terminal',
          type: 'tool-call'
        }
      ],
      metadata: {
        ...assistantMessage().metadata,
        custom: { timelineCompletedAt: landedAt, timelineTimestamp: startedAt }
      }
    } as unknown as ThreadMessage

    const { container } = render(<Harness assistant={assistant} />)
    await screen.findByText('Messaged badr')

    const assistantRow = container.querySelector('[data-slot="aui_assistant-message-root"]')
    const stamps = Array.from(assistantRow?.querySelectorAll('[data-slot="timeline-timestamp"]') ?? [])

    expect(stamps).toHaveLength(2)
    expect(stamps.map(stamp => stamp.textContent?.trim())).toEqual([
      formatClockTimestamp(startedAt),
      formatClockTimestamp(landedAt)
    ])
  })

  it('shows one clock while an outbound agent delivery is pending', async () => {
    const startedAt = createdAt.getTime() / 1000

    const assistant = {
      ...assistantMessage(),
      content: [
        {
          args: {
            command: 'hermes -p badr chat --in ~ -c "Bot Chat" -Q -q "Message from 🤖 Abu Saud: review"'
          },
          argsText: '{}',
          timestamp: startedAt,
          toolCallId: 'delivery-pending',
          toolName: 'terminal',
          type: 'tool-call'
        }
      ],
      status: { type: 'running' },
      metadata: {
        ...assistantMessage().metadata,
        custom: { timelineTimestamp: startedAt }
      }
    } as unknown as ThreadMessage

    const { container } = render(<Harness assistant={assistant} />)
    await screen.findByText(/Messaging badr/)

    const assistantRow = container.querySelector('[data-slot="aui_assistant-message-root"]')
    const stamps = Array.from(assistantRow?.querySelectorAll('[data-slot="timeline-timestamp"]') ?? [])

    expect(stamps).toHaveLength(1)
    expect(stamps[0]?.textContent?.trim()).toBe(formatClockTimestamp(startedAt))
  })

  it('keeps the aggregate prose clock beside delivery notice clocks', async () => {
    const startedAt = createdAt.getTime() / 1000
    const landedAt = startedAt + 130

    const assistant = {
      ...assistantMessage(),
      content: [
        { completedAt: startedAt + 1, text: 'I sent this for review.', timestamp: startedAt + 0.5, type: 'text' },
        {
          args: {
            command: 'hermes -p badr chat --in ~ -c "Bot Chat" -Q -q "Message from 🤖 Abu Saud: review"'
          },
          argsText: '{}',
          completedAt: landedAt,
          result: { exit_code: 0, output: 'reviewed' },
          timestamp: startedAt + 1,
          toolCallId: 'delivery-with-prose',
          toolName: 'terminal',
          type: 'tool-call'
        }
      ],
      metadata: {
        ...assistantMessage().metadata,
        custom: { timelineCompletedAt: landedAt, timelineTimestamp: startedAt }
      }
    } as unknown as ThreadMessage

    const { container } = render(<Harness assistant={assistant} />)
    await screen.findByText('I sent this for review.')
    await screen.findByText('Messaged badr')

    const assistantRow = container.querySelector('[data-slot="aui_assistant-message-root"]')
    const stamps = Array.from(assistantRow?.querySelectorAll('[data-slot="timeline-timestamp"]') ?? [])

    expect(stamps).toHaveLength(3)
    expect(stamps.filter(stamp => stamp.textContent?.trim() === formatClockTimestamp(landedAt))).toHaveLength(2)
  })

  it('keeps the aggregate clock when an agent delivery tool fails', async () => {
    const startedAt = createdAt.getTime() / 1000
    const landedAt = startedAt + 130

    const assistant = {
      ...assistantMessage(),
      content: [
        {
          args: {
            command: 'hermes -p badr chat --in ~ -c "Bot Chat" -Q -q "Message from 🤖 Abu Saud: review"'
          },
          argsText: '{}',
          completedAt: landedAt,
          isError: true,
          result: { error: 'delivery failed' },
          timestamp: startedAt,
          toolCallId: 'delivery-failed',
          toolName: 'terminal',
          type: 'tool-call'
        }
      ],
      metadata: {
        ...assistantMessage().metadata,
        custom: { timelineCompletedAt: landedAt, timelineTimestamp: startedAt }
      }
    } as unknown as ThreadMessage

    const { container } = render(<Harness assistant={assistant} />)
    await screen.findByText(/Ran hermes -p badr/)

    const assistantRow = container.querySelector('[data-slot="aui_assistant-message-root"]')
    const stamps = Array.from(assistantRow?.querySelectorAll('[data-slot="timeline-timestamp"]') ?? [])

    expect(stamps.filter(stamp => stamp.textContent?.trim() === formatClockTimestamp(landedAt))).toHaveLength(1)
    expect(stamps.map(stamp => stamp.textContent?.trim())).toContain(formatTimelineRange(startedAt, landedAt))
  })

  it('shows one clock on a settled inter-agent reply notice', async () => {
    const startedAt = createdAt.getTime() / 1000
    const landedAt = startedAt + 130

    const user = {
      ...userMessage(),
      content: [{ type: 'text', text: 'Message from 🤖 Badr (@badr): review this' }]
    } as unknown as ThreadMessage

    const assistant = {
      ...assistantMessage(),
      content: [{ completedAt: landedAt, text: 'reviewed', timestamp: startedAt, type: 'text' }],
      metadata: {
        ...assistantMessage().metadata,
        custom: { timelineCompletedAt: landedAt, timelineTimestamp: startedAt }
      }
    } as unknown as ThreadMessage

    const { container } = render(<Harness assistant={assistant} user={user} />)
    await screen.findByText('Replied to Badr')

    const assistantRow = container.querySelector('[data-slot="aui_assistant-message-root"]')
    const stamps = Array.from(assistantRow?.querySelectorAll('[data-slot="timeline-timestamp"]') ?? [])

    expect(stamps).toHaveLength(1)
    expect(stamps[0]?.textContent?.trim()).toBe(formatClockTimestamp(landedAt))
  })
})
