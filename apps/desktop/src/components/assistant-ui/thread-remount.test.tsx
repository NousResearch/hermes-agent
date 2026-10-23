import { AssistantRuntimeProvider, type ThreadMessage, useExternalStoreRuntime } from '@assistant-ui/react'
import { act, render, screen, waitFor } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import { createdAt, stubThreadEnvironment, stubThreadViewportSize } from './test-utils'
import { Thread } from './thread'

stubThreadEnvironment()
stubThreadViewportSize()

const MESSAGES: ThreadMessage[] = [
  {
    id: 'user-1',
    role: 'user',
    content: [{ type: 'text', text: 'hello from the user' }],
    attachments: [],
    createdAt,
    metadata: { custom: {} }
  } as ThreadMessage,
  {
    id: 'assistant-1',
    role: 'assistant',
    content: [{ type: 'text', text: 'stable assistant reply' }],
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
]

function Harness({
  onBranchInNewChat,
  onCancel
}: {
  onBranchInNewChat: (messageId: string) => void
  onCancel: () => void
}) {
  const runtime = useExternalStoreRuntime<ThreadMessage>({
    messages: MESSAGES,
    isRunning: false,
    onNew: async () => {}
  })

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <Thread onBranchInNewChat={onBranchInNewChat} onCancel={onCancel} />
    </AssistantRuntimeProvider>
  )
}

describe('thread message mount stability', () => {
  // Regression: the desktop controller re-renders every 15s (status
  // snapshot poll) and used to pass freshly-created callbacks down to
  // <Thread/>. Those callbacks were deps of the `messageComponents`
  // useMemo, so new component *types* were created each poll and React
  // unmounted/remounted every visible message — shiki re-highlighted
  // code blocks and the whole thread visibly jumped.
  it('keeps message DOM nodes mounted when callback props get new identities', async () => {
    const { rerender } = render(<Harness onBranchInNewChat={() => {}} onCancel={() => {}} />)

    await waitFor(() => {
      expect(screen.getByText('stable assistant reply')).toBeTruthy()
      expect(screen.getByText('hello from the user')).toBeTruthy()
    })

    const assistantBefore = screen.getByText('stable assistant reply')
    const userBefore = screen.getByText('hello from the user')

    // Same data, new callback identities — exactly what a parent
    // re-render driven by an unrelated state update produces.
    await act(async () => {
      rerender(<Harness onBranchInNewChat={() => {}} onCancel={() => {}} />)
    })

    expect(screen.getByText('stable assistant reply')).toBe(assistantBefore)
    expect(screen.getByText('hello from the user')).toBe(userBefore)
  })
})
