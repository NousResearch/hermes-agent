import { MessagePrimitive, type ThreadMessage } from '@assistant-ui/react'
import { act, render, waitFor } from '@testing-library/react'
import { expect, it } from 'vitest'

import { PaneLifecycleContext, PaneVisibleContext } from '@/components/pane-shell/pane-visibility'
import { stubThreadEnvironment, stubThreadViewportSize, ThreadRuntime } from '../test-utils'
import { ThreadMessageList } from './list'

stubThreadEnvironment()
stubThreadViewportSize()

const Row = () => <MessagePrimitive.Root />
const components = { AssistantMessage: Row, UserMessage: Row }
const messages: ThreadMessage[] = Array.from({ length: 12 }, (_, i): ThreadMessage[] => [
  {
    id: `u${i}`,
    createdAt: new Date(0),
    role: 'user',
    content: [{ type: 'text', text: `prompt ${i}` }],
    attachments: [],
    metadata: { custom: {} }
  },
  {
    id: `a${i}`,
    createdAt: new Date(0),
    role: 'assistant',
    content: [{ type: 'text', text: 'long explanation '.repeat(1800) }],
    status: { type: 'complete', reason: 'stop' },
    metadata: { unstable_state: null, unstable_annotations: [], unstable_data: [], steps: [], custom: {} }
  }
]).flat()

it('does not spend the visible-page minimum on hidden heavy turns, and restores it on reveal', async () => {
  const pane = (visible: boolean) => (
    <PaneVisibleContext value={visible}>
      <PaneLifecycleContext value={visible ? 'visible' : 'hot-hidden'}>
        <ThreadRuntime messages={messages}>
          <ThreadMessageList clampToComposer={false} components={components} sessionKey="budget" />
        </ThreadRuntime>
      </PaneLifecycleContext>
    </PaneVisibleContext>
  )
  const view = render(pane(true))
  const rows = () => view.container.querySelectorAll('[data-slot="aui_message-group"]').length
  await waitFor(() => expect(rows()).toBeGreaterThanOrEqual(8))
  view.rerender(pane(false))
  await act(async () => {
    await new Promise(resolve => setTimeout(resolve, 50))
  })
  expect(rows()).toBeLessThanOrEqual(2)
  view.rerender(pane(true))
  await waitFor(() => expect(rows()).toBeGreaterThanOrEqual(8))
  expect(messages).toHaveLength(24)
})
