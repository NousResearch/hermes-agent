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

import { isOwnerDirectedAssistantText } from './assistant-message'
import { formatTimelineRange, formatTimelineTimestamp } from './timestamp'

import { Thread } from '.'

// Timeline timestamps render only when `display.timestamps` is enabled.
$displayTimestamps.set(true)

const createdAt = new Date('2026-05-01T00:00:00.000Z')
const completedAt = createdAt.getTime() / 1000 + 1.25
stubThreadEnvironment()

afterEach(() => {
  cleanup()
})

function userMessage(content = 'question one'): ThreadMessage {
  return {
    id: 'user-1',
    role: 'user',
    content: [{ type: 'text', text: content }],
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
  user = userMessage(),
  onBranchInNewChat
}: {
  assistant?: ThreadMessage
  user?: ThreadMessage
  onBranchInNewChat?: (messageId: string) => void
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

describe('inter-agent assistant audience handling', () => {
  it('recognizes owner-directed report markers', () => {
    expect(isOwnerDirectedAssistantText('【给 Kuro 的独立汇报】\nDone.')).toBe(true)
    expect(isOwnerDirectedAssistantText('[for you]\nDone.')).toBe(true)
    expect(isOwnerDirectedAssistantText('For you: Done.')).toBe(true)
    expect(isOwnerDirectedAssistantText('Ordinary reply to the other bot.')).toBe(false)
  })

  it('keeps owner-directed reports visible instead of collapsing them under the triggering bot', async () => {
    render(
      <Harness
        assistant={
          {
            ...assistantMessage(),
            content: [{ text: '【给 Kuro 的独立汇报】\nOwner-visible report.', type: 'text' }]
          } as unknown as ThreadMessage
        }
        user={userMessage('Message from 🤖 reviewer (@reviewer): PASS relay')}
      />
    )

    expect(await screen.findByText('For you')).toBeTruthy()
    expect(await screen.findByText(/Owner-visible report/)).toBeTruthy()
    expect(screen.queryByText('Replied to reviewer')).toBeNull()
  })

  it('still collapses ordinary inter-agent replies', async () => {
    render(
      <Harness
        assistant={
          {
            ...assistantMessage(),
            content: [{ text: 'ACK, reviewer.', type: 'text' }]
          } as unknown as ThreadMessage
        }
        user={userMessage('Message from 🤖 reviewer (@reviewer): PASS relay')}
      />
    )

    expect(await screen.findByText('Replied to reviewer')).toBeTruthy()
    expect(screen.queryByText('For you')).toBeNull()
  })
})

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
  it('always renders precise user and assistant lifecycle times', async () => {
    const { container } = render(<Harness />)

    await screen.findByText('done')

    const stamps = Array.from(container.querySelectorAll('[data-slot="timeline-timestamp"]')).map(node =>
      node.textContent?.trim()
    )

    const startedAt = createdAt.getTime() / 1000

    expect(stamps).toContain(formatTimelineTimestamp(startedAt))
    expect(stamps).toContain(formatTimelineRange(startedAt, completedAt))
    expect(stamps).toContain(formatTimelineRange(startedAt + 0.05, startedAt + 0.1))
    expect(stamps).toContain(formatTimelineRange(startedAt + 0.125, startedAt + 0.5))
  })

  it('suppresses an aggregate assistant stamp that exactly duplicates its sole part', async () => {
    const startedAt = createdAt.getTime() / 1000

    const assistant = {
      ...assistantMessage(),
      content: [{ completedAt, text: 'done', timestamp: startedAt, type: 'text' }]
    } as unknown as ThreadMessage

    const { container } = render(<Harness assistant={assistant} />)

    await screen.findByText('done')

    const stamps = Array.from(container.querySelectorAll('[data-slot="timeline-timestamp"]')).map(node =>
      node.textContent?.trim()
    )

    expect(stamps.filter(stamp => stamp === formatTimelineRange(startedAt, completedAt))).toHaveLength(1)
  })
})
