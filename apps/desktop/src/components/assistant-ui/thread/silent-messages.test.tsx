import { AssistantRuntimeProvider, type ThreadMessage, useExternalStoreRuntime } from '@assistant-ui/react'
import { cleanup, fireEvent, render, waitFor } from '@testing-library/react'
import { useState } from 'react'
import { afterEach, describe, expect, it } from 'vitest'

import { useRuntimeMessageRepository } from '@/app/chat/runtime-repository'
import { type ChatMessage, toChatMessages } from '@/lib/chat-messages'
import type { SessionMessage } from '@/types/hermes'

import { stubThreadEnvironment } from '../test-utils'

import { Thread } from '.'

stubThreadEnvironment()
afterEach(cleanup)

function Harness({ messages }: { messages: ChatMessage[] }) {
  const repository = useRuntimeMessageRepository(messages)
  const [headId, setHeadId] = useState<string | null>(null)
  const runtime = useExternalStoreRuntime<ThreadMessage>({
    messageRepository: headId ? { ...repository, headId } : repository,
    setMessages: next => setHeadId(next.at(-1)?.id ?? null),
    isRunning: messages.some(message => Boolean(message.pending)),
    onNew: async () => {}
  })

  return <AssistantRuntimeProvider runtime={runtime}><Thread /></AssistantRuntimeProvider>
}

const assistant = (body: string, pending = false): ChatMessage => ({
  id: 'reply', role: 'assistant', parts: [{ type: 'text', text: body }], pending
})

describe('quiet internal transcript', () => {
  it('keeps previous answers reachable when the selected alternative is silent', async () => {
    const view = render(<Harness messages={[
      { id: 'user', role: 'user', parts: [{ type: 'text', text: 'Check' }] },
      { id: 'answer', role: 'assistant', branchGroupId: 'alternatives', parts: [{ type: 'text', text: 'Previous useful answer' }] },
      { id: 'silent', role: 'assistant', branchGroupId: 'alternatives', parts: [{ type: 'text', text: '[[SILENT]]' }] },
    ]} />)
    const counter = await view.findByText('2 / 2')
    expect(view.container.textContent).not.toContain('[[SILENT]]')
    fireEvent.click(counter.parentElement!.querySelector('button')!)
    await view.findByText('Previous useful answer')
  })

  it('suppresses only silent assistant responses through streaming, settlement and hydration without empty bubbles', async () => {
    const { container, rerender } = render(<Harness messages={[assistant('[' , true)]} />)
    for (const body of ['[', '[[SI', '[[SILENT]', '[[SILENT]]', ' \n[[SILENT]]\n']) {
      rerender(<Harness messages={[assistant(body, true)]} />)
      expect(container.querySelector('[data-role="assistant"]')).toBeNull()
    }
    rerender(<Harness messages={[assistant('[[SILENT]]')]} />)
    expect(container.querySelector('[data-role="assistant"]')).toBeNull()

    const rows: SessionMessage[] = [
      { role: 'user', content: 'Check the job', timestamp: 1 },
      { role: 'assistant', content: 'Useful progress', timestamp: 2 },
      { role: 'assistant', content: '[[SILENT]]', timestamp: 3 }
    ]
    const original = structuredClone(rows)
    rerender(<Harness messages={toChatMessages(rows)} />)
    await waitFor(() => expect(container.textContent).toContain('Useful progress'))
    expect(container.textContent).not.toContain('[[SILENT]]')
    expect(rows).toEqual(original)

    for (const body of ['[[SILENT]] is a control token', '[[Something else]]', '[']) {
      rerender(<Harness messages={[assistant(body)]} />)
      await waitFor(() => expect(container.textContent).toContain(body))
    }
    rerender(<Harness messages={[{ ...assistant('[[SILENT]]'), role: 'user' }]} />)
    await waitFor(() => expect(container.textContent).toContain('[[SILENT]]'))
    rerender(<Harness messages={[{ ...assistant('[[SILENT]]'), error: 'Important failure' }]} />)
    await waitFor(() => expect(container.textContent).toContain('Important failure'))
  })

  it('folds trusted internal events but never classifies human text by its envelope', async () => {
    const content = '[ASYNC DELEGATION COMPLETE — test]\nPrivate instructions\n--- RESULT ---\nReport retained'
    const internal: SessionMessage = {
      role: 'user', content, timestamp: 2, display_kind: 'internal_event',
      display_metadata: { user_originated: false, event_kind: 'workflow.async_delegation.terminal', task_count: 1 }
    }
    const rows: SessionMessage[] = [
      { role: 'user', content: 'Check the job', timestamp: 1 }, internal,
      { role: 'assistant', content: 'Actionable conclusion', timestamp: 3 }
    ]
    const original = structuredClone(rows)
    const { container, getByRole, rerender } = render(<Harness messages={toChatMessages(rows)} />)
    expect(container.textContent).not.toContain('Private instructions')
    expect(container.textContent).not.toContain('Report retained')
    expect(container.textContent).toContain('Actionable conclusion')
    fireEvent.click(getByRole('button', { name: '1 background agent finished' }))
    expect(container.textContent).toContain('Report retained')
    expect(rows).toEqual(original)

    rerender(<Harness messages={toChatMessages([{ role: 'user', content, timestamp: 4 }])} />)
    await waitFor(() => expect(container.textContent).toContain('Private instructions'))
    rerender(<Harness messages={toChatMessages([{ ...internal, timestamp: 5, display_metadata: { user_originated: true } }])} />)
    await waitFor(() => expect(container.textContent).toContain('Private instructions'))
  })
})
