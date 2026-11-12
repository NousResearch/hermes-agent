import { AssistantRuntimeProvider, type ThreadMessage, useExternalStoreRuntime } from '@assistant-ui/react'
import { cleanup, render } from '@testing-library/react'
import { afterEach, expect, it } from 'vitest'

import { en } from '@/i18n/en'

import { stubThreadEnvironment } from '../test-utils'

import { Thread } from '.'

const timestamp = new Date('2026-05-01T00:00:00.000Z')
stubThreadEnvironment()

// A safety-refused turn arrives (post-hydration) as a `system` row carrying the
// user's own text plus `metadata.custom.contentRefused`. It must render as the
// greyed, non-interactive refused block — never a blue user bubble.
function Harness({ userText, notice }: { userText: string; notice?: string }) {
  const message = {
    id: 'content-refused-1',
    role: 'system',
    content: [{ type: 'text', text: userText }],
    createdAt: timestamp,
    metadata: { custom: { contentRefused: { ...(notice ? { notice } : {}) } } }
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

it('renders the refused block with the user text and the backend notice, not a user bubble', () => {
  const notice = 'Safety filter refused this — ask something else.'
  const { container } = render(<Harness notice={notice} userText="the request the filter blocked" />)

  const block = container.querySelector('[data-slot="aui_content-refused"]')
  expect(block).toBeTruthy()
  expect(block?.textContent).toContain('the request the filter blocked')
  expect(block?.textContent).toContain(notice)
  // It is NOT a user bubble and NOT an assistant reply.
  expect(container.querySelector('[data-role="user"]')).toBeNull()
  expect(container.querySelector('[data-slot="aui_user-message-root"]')).toBeNull()
})

it('falls back to the i18n notice when the backend sent none', () => {
  const { container } = render(<Harness userText="blocked text" />)

  const block = container.querySelector('[data-slot="aui_content-refused"]')
  expect(block).toBeTruthy()
  expect(block?.textContent).toContain('blocked text')
  expect(block?.textContent).toContain(en.assistant.thread.contentRefusedNotice)
})
