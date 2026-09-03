import { AssistantRuntimeProvider, type ThreadMessage, useExternalStoreRuntime } from '@assistant-ui/react'
import { cleanup, render } from '@testing-library/react'
import { afterEach, expect, it } from 'vitest'

import { $displayTimestamps } from '@/store/display-timestamps'

import { stubThreadEnvironment } from '../test-utils'

import { Thread } from '.'

$displayTimestamps.set(true)
stubThreadEnvironment()

const createdAt = new Date('2026-05-01T00:00:00.000Z')

function Harness() {
  const message = {
    id: 'agent-delivery-1',
    role: 'user',
    content: [{ type: 'text', text: 'Message from 🤖 Badr (@badr): review complete' }],
    attachments: [],
    createdAt,
    metadata: { custom: { timelineTimestamp: createdAt.getTime() / 1000 } }
  } as unknown as ThreadMessage

  const runtime = useExternalStoreRuntime<ThreadMessage>({
    messages: [message],
    isRunning: false,
    onNew: async () => {}
  })

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <Thread />
    </AssistantRuntimeProvider>
  )
}

afterEach(cleanup)

it('shows a send clock on an agent delivery notice', () => {
  const { container } = render(<Harness />)
  const row = container.querySelector('[data-slot="aui_user-message-root"]')

  expect(row?.querySelector('[data-slot="timeline-timestamp"]')).toBeTruthy()
})
