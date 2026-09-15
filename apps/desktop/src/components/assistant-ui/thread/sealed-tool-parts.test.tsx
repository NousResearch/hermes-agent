import { type ThreadMessage } from '@assistant-ui/react'
import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { $activeSessionId } from '@/store/session'

import { stubThreadEnvironment, stubThreadViewportSize, ThreadRuntime } from '../test-utils'
import { Thread } from '../thread'

// The image card, delegate list and delivery notice decide pending-ness from
// `result === undefined`; a call sealed without a result must not stay live.

stubThreadEnvironment()
stubThreadViewportSize()

const createdAt = new Date('2026-09-11T12:00:00Z')
const sealedAt = createdAt.getTime() / 1000 + 5

function sealedMessage(toolName: string, args: Record<string, unknown>): ThreadMessage {
  return {
    id: `assistant-sealed-${toolName}`,
    role: 'assistant',
    content: [
      {
        type: 'tool-call',
        toolCallId: `${toolName}-1`,
        toolName,
        args,
        argsText: JSON.stringify(args),
        completedAt: sealedAt
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

const Harness = ({ message }: { message: ThreadMessage }) => (
  <ThreadRuntime messages={[message]}>
    <Thread />
  </ThreadRuntime>
)

afterEach(() => {
  cleanup()
  $activeSessionId.set(null)
})

describe('tool parts sealed without a result', () => {
  it('renders a sealed delegate_task as the generic row, not a running task list', async () => {
    $activeSessionId.set('sess-1')
    const { container } = render(<Harness message={sealedMessage('delegate_task', { tasks: [{ goal: 'inspect' }] })} />)

    expect(await screen.findByText('Result unavailable')).toBeTruthy()
    expect(container.querySelector('[data-delegate-card]')).toBeNull()
    expect(screen.queryByLabelText('Running')).toBeNull()
  })

  it('renders a sealed image_generate as the generic row, not a rendering placeholder', async () => {
    const { container } = render(<Harness message={sealedMessage('image_generate', { prompt: 'a cat' })} />)

    expect(await screen.findByText('Result unavailable')).toBeTruthy()
    expect(container.querySelector('[data-slot="aui_generated-image"]')).toBeNull()
    expect(screen.queryByRole('status')).toBeNull()
  })

  it('renders a sealed delivery terminal call as the generic row, not a pending notice', async () => {
    const command = 'hermes -p turqoise chat --in ~ -c "Bot Chat" -Q -q "Message from 🤖 Hermes (@hermes): hi"'
    const { container } = render(<Harness message={sealedMessage('terminal', { command })} />)

    expect(await screen.findByText('Result unavailable')).toBeTruthy()
    expect(container.querySelector('[data-slot="aui_agent-delivery-notice"]')).toBeNull()
  })
})
