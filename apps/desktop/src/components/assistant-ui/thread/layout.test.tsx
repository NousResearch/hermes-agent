import { AssistantRuntimeProvider, type ThreadMessage, useExternalStoreRuntime } from '@assistant-ui/react'
import { cleanup, render } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { stubThreadEnvironment, stubThreadViewportSize } from '../test-utils'

import { Thread } from '.'

stubThreadEnvironment()
stubThreadViewportSize()

afterEach(cleanup)

const createdAt = new Date('2026-09-13T00:00:00.000Z')

function transcript(): ThreadMessage[] {
  return [
    {
      id: 'user-layout',
      role: 'user',
      content: [{ type: 'text', text: 'Show the layout.' }],
      attachments: [],
      createdAt,
      metadata: { custom: {} }
    } as ThreadMessage,
    {
      id: 'assistant-layout',
      role: 'assistant',
      content: [{ type: 'text', text: 'Centered assistant markdown.' }],
      status: { type: 'complete', reason: 'stop' },
      createdAt,
      metadata: { unstable_state: null, unstable_annotations: [], unstable_data: [], steps: [], custom: {} }
    } as ThreadMessage
  ]
}

function Harness({ messages }: { messages: ThreadMessage[] }) {
  const runtime = useExternalStoreRuntime<ThreadMessage>({ isRunning: false, messages, onNew: async () => {} })

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <Thread sessionKey="layout" />
    </AssistantRuntimeProvider>
  )
}

function expectReadingColumn(container: HTMLElement) {
  const content = container.querySelector<HTMLElement>('[data-slot="aui_thread-content"]')

  expect(content).not.toBeNull()
  expect(content?.className).toContain('max-w-(--conversation-width)')
  expect(content?.className).not.toContain('max-w-(--composer-width)')
}

describe('thread reading column', () => {
  it('uses the conversation width for populated and empty transcripts', () => {
    const view = render(<Harness messages={transcript()} />)

    expectReadingColumn(view.container)

    view.rerender(<Harness messages={[]} />)
    expectReadingColumn(view.container)
  })
})
