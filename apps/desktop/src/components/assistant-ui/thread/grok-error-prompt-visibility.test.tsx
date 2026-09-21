import { AssistantRuntimeProvider, type ThreadMessage, useExternalStoreRuntime } from '@assistant-ui/react'
import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import { stubThreadEnvironment, stubThreadViewportSize } from '../test-utils'

import { Thread } from '.'

stubThreadEnvironment()
stubThreadViewportSize()

const createdAt = new Date('2026-05-01T00:00:00.000Z')
const prompt = 'Grok please answer this submitted prompt'

function userMsg(): ThreadMessage {
  return {
    id: 'user-1',
    role: 'user',
    content: [{ type: 'text', text: prompt }],
    attachments: [],
    createdAt,
    metadata: { custom: {} }
  } as ThreadMessage
}

function errMsg(): ThreadMessage {
  return {
    id: 'assistant-error-1',
    role: 'assistant',
    content: [],
    status: { type: 'incomplete', reason: 'error', error: 'API call failed after 3 retries: capacity' },
    createdAt,
    metadata: { unstable_state: null, unstable_annotations: [], unstable_data: [], steps: [], custom: {} }
  } as ThreadMessage
}

function Harness() {
  const runtime = useExternalStoreRuntime<ThreadMessage>({
    messages: [userMsg(), errMsg()],
    isRunning: false,
    onNew: async () => {}
  })

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <Thread />
    </AssistantRuntimeProvider>
  )
}

describe('grok error keeps user prompt painted (#101310)', () => {
  it('keeps the submitted prompt readable without hover after a Grok-shaped error', () => {
    const { container } = render(<Harness />)

    expect(screen.getByText(prompt)).toBeTruthy()

    const root = container.querySelector('[data-slot="aui_user-message-root"]') as HTMLElement
    expect(root).toBeTruthy()
    expect(root.textContent).toContain(prompt)

    const bubble = container.querySelector('.composer-human-message') as HTMLElement
    expect(bubble).toBeTruthy()
    expect(bubble.textContent).toContain(prompt)

    const actions = container.querySelector('[data-slot="aui_user-bubble-actions"]') as HTMLElement
    expect(actions).toBeTruthy()
    expect(actions.textContent).toContain(prompt)

    const textEl = bubble.querySelector('.wrap-anywhere') as HTMLElement | null
    expect(textEl?.textContent).toContain(prompt)
    expect(textEl?.className ?? '').not.toMatch(/opacity-0/)
  })
})
