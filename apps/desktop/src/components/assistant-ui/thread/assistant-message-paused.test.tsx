// While its chat is paused for an automatic continuation (store/paused-replies.ts),
// the newest reply reads "Continuing…" instead of "Response stopped": the turn
// was stopped only so its interrupter could apply a change and go on with it.
import { AssistantRuntimeProvider, type ThreadMessage, useExternalStoreRuntime } from '@assistant-ui/react'
import { act, cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { en } from '@/i18n/en'
import { continueReplyPause, endReplyPause, markReplyPaused } from '@/store/paused-replies'
import { $selectedStoredSessionId } from '@/store/session'

import { stubThreadEnvironment } from '../test-utils'

import { Thread } from '.'

const CHAT = 'paused-thread-chat'
const createdAt = new Date('2026-11-08T23:04:15.000Z')

stubThreadEnvironment()

const prompt = {
  id: 'user-1',
  role: 'user',
  content: [{ type: 'text', text: 'and now read DELIVERY-PLAN.md' }],
  attachments: [],
  createdAt,
  metadata: { custom: {} }
} as unknown as ThreadMessage

// The paused turn's last bubble as the stored read after the interrupted
// completion leaves it: its closing row lends it the stopped mark.
const stoppedReply = {
  id: 'assistant-1',
  role: 'assistant',
  content: [{ type: 'text', text: 'Reading the plan.' }],
  status: { type: 'complete', reason: 'stop' },
  createdAt,
  metadata: { unstable_state: null, unstable_annotations: [], unstable_data: [], steps: [], custom: { interrupted: true } }
} as unknown as ThreadMessage

// The continuation's first bubble, sealed at its first tool call (message.interim) while the turn runs on.
const continuationInterim = {
  ...stoppedReply,
  id: 'assistant-2',
  content: [{ type: 'text', text: 'Now I can read it.' }],
  metadata: { ...stoppedReply.metadata, custom: { interim: true } }
} as unknown as ThreadMessage

function Harness({ messages = [prompt, stoppedReply] }: { messages?: ThreadMessage[] }) {
  const runtime = useExternalStoreRuntime<ThreadMessage>({
    messages,
    isRunning: false,
    onNew: async () => {}
  })

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <Thread />
    </AssistantRuntimeProvider>
  )
}

describe('the mark of a reply paused for an automatic continuation', () => {
  afterEach(() => {
    cleanup()
    endReplyPause(CHAT)
    $selectedStoredSessionId.set(null)
  })

  it('reads "Continuing…" while the chat is paused, and "Response stopped" once the pause ends unfollowed', async () => {
    $selectedStoredSessionId.set(CHAT)
    markReplyPaused(CHAT)
    render(<Harness />)

    expect(await screen.findByText(en.assistant.thread.responseContinuing)).toBeTruthy()
    expect(screen.queryByText(en.assistant.thread.responseStopped)).toBeNull()

    act(() => endReplyPause(CHAT))

    expect(await screen.findByText(en.assistant.thread.responseStopped)).toBeTruthy()
    expect(screen.queryByText(en.assistant.thread.responseContinuing)).toBeNull()
  })

  it('reads neither once the continuation has started: the reply goes on', async () => {
    $selectedStoredSessionId.set(CHAT)
    markReplyPaused(CHAT)
    act(() => continueReplyPause(CHAT))
    const continuedReply = { ...stoppedReply, metadata: { ...stoppedReply.metadata, custom: {} } } as ThreadMessage
    render(<Harness messages={[prompt, continuedReply, continuationInterim]} />)

    expect(await screen.findByText('Now I can read it.')).toBeTruthy()
    expect(screen.queryByText(en.assistant.thread.responseContinuing)).toBeNull()
    expect(screen.queryByText(en.assistant.thread.responseStopped)).toBeNull()
  })

  it('leaves another chat\'s stopped reply alone', async () => {
    $selectedStoredSessionId.set(CHAT)
    markReplyPaused('some-other-chat')

    try {
      render(<Harness />)

      expect(await screen.findByText(en.assistant.thread.responseStopped)).toBeTruthy()
    } finally {
      endReplyPause('some-other-chat')
    }
  })
})
