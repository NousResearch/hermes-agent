// The thinking indicator (dither block) may only ever render at the TAIL of
// the thread. A message stuck status:running mid-transcript — however it got
// there (missed settle event, steer race, upstream state bug) — must render
// its content with no spinner: a live indicator above a later user message
// reads as the agent answering out of order.
import { type ThreadMessage } from '@assistant-ui/react'
import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { stubThreadEnvironment, ThreadRuntime, userMessage } from '../test-utils'

import { Thread } from '.'

const createdAt = new Date('2026-05-01T00:00:00.000Z')
stubThreadEnvironment()

afterEach(() => {
  cleanup()
})

const assistantMetadata = {
  unstable_state: null,
  unstable_annotations: [],
  unstable_data: [],
  steps: [],
  custom: {}
}

function assistant(id: string, text: string, running: boolean): ThreadMessage {
  return {
    id,
    role: 'assistant',
    content: text ? [{ type: 'text', text }] : [],
    status: running ? { type: 'running' } : { type: 'complete', reason: 'stop' },
    createdAt,
    metadata: assistantMetadata
  } as ThreadMessage
}

const Harness = ({ messages }: { messages: ThreadMessage[] }) => (
  <ThreadRuntime messages={messages}>
    <Thread />
  </ThreadRuntime>
)

describe('thinking indicator is tail-only', () => {
  it('shows the loading indicator on a running placeholder at the tail', async () => {
    const { container } = render(<Harness messages={[userMessage('u1', 'question'), assistant('a1', '', true)]} />)

    expect(await screen.findByRole('status', { name: 'Hermes is loading a response' })).toBeTruthy()
    expect(container.querySelector('[data-slot="aui_response-loading"]')).toBeTruthy()
  })

  it('never shows an indicator on a stale running message mid-transcript', async () => {
    // A stranded pending bubble from an earlier turn, then a newer exchange.
    const { container } = render(
      <Harness
        messages={[
          userMessage('u1', 'first question'),
          assistant('a1', '', true),
          userMessage('u2', 'second question'),
          assistant('a2', 'answered', false)
        ]}
      />
    )

    await screen.findByText('answered')

    expect(container.querySelector('[data-slot="aui_response-loading"]')).toBeNull()
    expect(container.querySelector('[data-slot="aui_turn-activity"]')).toBeNull()
  })
})
