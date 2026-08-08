// Bug #2: the Branch-in-new-chat button used to render unconditionally even
// when its handler was a no-op (session-tile.tsx passed `() => undefined`
// for branched/tiled chats, where nested branching isn't supported). That
// left a visibly clickable button that silently did nothing. The fix makes
// AssistantMessage's action bar hide the button entirely when no handler is
// supplied, matching how onDismissError/onRestoreToMessage already behave.
import { AssistantRuntimeProvider, type ThreadMessage, useExternalStoreRuntime } from '@assistant-ui/react'
import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { toChatMessages } from '@/lib/chat-messages'
import { toRuntimeMessage } from '@/lib/chat-runtime'

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

function userMessage(): ThreadMessage {
  return {
    id: 'user-1',
    role: 'user',
    content: [{ type: 'text', text: 'question one' }],
    attachments: [],
    createdAt,
    metadata: { custom: {} }
  } as ThreadMessage
}

function assistantMessage(turnDurationSeconds?: number, text = 'done'): ThreadMessage {
  const [message] = toChatMessages([
    {
      role: 'assistant',
      content: text,
      timestamp: createdAt.getTime() / 1000,
      ...(text
        ? {}
        : {
            tool_calls: [
              {
                id: 'call-1',
                type: 'function',
                function: { name: 'terminal', arguments: '{}' }
              }
            ]
          }),
      ...(turnDurationSeconds === undefined ? {} : { display_metadata: { turn_duration_seconds: turnDurationSeconds } })
    }
  ])

  return toRuntimeMessage(message)
}

function Harness({
  assistantText = 'done',
  onBranchInNewChat,
  turnDurationSeconds
}: {
  assistantText?: string
  onBranchInNewChat?: (messageId: string) => void
  turnDurationSeconds?: number
}) {
  const runtime = useExternalStoreRuntime<ThreadMessage>({
    messages: [userMessage(), assistantMessage(turnDurationSeconds, assistantText)],
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

describe('AssistantMessage turn duration', () => {
  it('shows the completed duration in the message action bar', async () => {
    render(<Harness turnDurationSeconds={65.4} />)

    const duration = await screen.findByText('1:05')
    expect(duration.getAttribute('data-slot')).toBe('aui_msg-turn-duration')
    expect(duration.getAttribute('aria-label')).toBe('1m 5s')
  })

  it('shows the completed duration when the message has no text', async () => {
    render(<Harness assistantText="" turnDurationSeconds={12.4} />)

    expect(await screen.findByText('12s')).not.toBeNull()
  })
})
