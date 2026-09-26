import type { ThreadMessage } from '@assistant-ui/react'
import { cleanup, fireEvent, render, waitFor, within } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { toChatMessages } from '@/lib/chat-messages'
import { toRuntimeMessage } from '@/lib/chat-runtime'
import type { SessionMessage } from '@/types/hermes'

import { createdAt, stubThreadEnvironment, ThreadRuntime } from '../test-utils'

import { Thread } from '.'

beforeEach(stubThreadEnvironment)
afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

it('keeps background continuations in one response with one action bar and the original message identities', async () => {
  const content = '[IMPORTANT: Background process proc_example completed normally (exit code 0).\nOutput:\nVerified.]'
  const clipboard = { writeText: vi.fn().mockResolvedValue(undefined) }
  vi.stubGlobal('navigator', { ...navigator, clipboard })
  const branch = vi.fn()

  for (const display of [
    {},
    { display_kind: 'process_complete', display_metadata: { display_text: 'Background Process Finished: verify' } },
    { display_kind: 'async_delegation_complete', display_metadata: { display_text: 'Background agent finished' } }
  ]) {
    const stored = [
      { role: 'user', content: 'Verify it.', timestamp: 1 },
      { role: 'assistant', content: 'Checking the deployment.', timestamp: 2 },
      { role: 'user', content, timestamp: 3, ...display },
      { role: 'assistant', content: 'The deployment is verified.', timestamp: 4 }
    ] as SessionMessage[]

    const messages = toChatMessages(stored).map(toRuntimeMessage)

    const { container, rerender, unmount } = render(
      <ThreadRuntime messages={messages.slice(0, 2)}>
        <Thread onBranchInNewChat={branch} />
      </ThreadRuntime>
    )

    const original = container.querySelector('[data-role="assistant"]')

    rerender(
      <ThreadRuntime messages={messages}>
        <Thread onBranchInNewChat={branch} />
      </ThreadRuntime>
    )
    await waitFor(() => expect(container.textContent).toContain('The deployment is verified.'))
    expect(container.querySelectorAll('[data-slot="aui_turn-pair"]')).toHaveLength(1)
    expect(container.querySelectorAll('[data-slot="aui_response-group"]')).toHaveLength(1)
    expect(container.querySelectorAll('[data-slot="aui_msg-actions"]')).toHaveLength(1)
    expect(container.querySelector('[data-role="assistant"]')).toBe(original)
    expect(
      [...container.querySelectorAll('[data-role="assistant"]')].map(e => e.getAttribute('data-message-id'))
    ).toEqual([messages[1]!.id, messages[3]!.id])
    const actions = container.querySelector('[data-slot="aui_msg-actions"]') as HTMLElement
    fireEvent.click(within(actions).getByRole('button', { name: 'Copy' }))
    await waitFor(() =>
      expect(clipboard.writeText).toHaveBeenLastCalledWith('Checking the deployment.\n\nThe deployment is verified.')
    )
    fireEvent.click(within(actions).getByRole('button', { name: /branch/i }))
    expect(branch).toHaveBeenLastCalledWith(messages[3]!.id)

    const pending: ThreadMessage[] = [
      ...messages,
      { ...messages[3]!, role: 'assistant', id: 'next', content: [], status: { type: 'running' } } as ThreadMessage
    ]

    rerender(
      <ThreadRuntime messages={pending}>
        <Thread onBranchInNewChat={branch} />
      </ThreadRuntime>
    )
    await waitFor(() => expect(container.querySelectorAll('[data-role="assistant"]')).toHaveLength(3))
    expect(container.querySelectorAll('[data-slot="aui_msg-actions"]')).toHaveLength(1)

    unmount()

    const reloaded = render(
      <ThreadRuntime messages={messages}>
        <Thread />
      </ThreadRuntime>
    )

    expect(reloaded.container.querySelectorAll('[data-slot="aui_msg-actions"]')).toHaveLength(1)
    reloaded.unmount()
  }
})

// A REST transcript re-hydration rebuilds the tail through `toChatMessages`,
// whose ids are synthesized POSITIONALLY (`${timestamp}-${index}-${role}`), so a
// row can come back with a different id at the same index inside an
// already-mounted turn. The turn itself is keyed on the user message id, so it
// does not remount — which is exactly when an index-keyed child inherits the
// previous message's mounted state (assistant-ui's `PartByIndexProvider` keeps a
// per-index `lastPartRef`, and `MessageByIndex` memoizes on `index` alone).
// This asserts the row remounts. The stronger claim — that the previous
// message's PARTS were visible — is NOT proven here: `MessagePrimitiveParts`
// derives its ranges from the current message's parts, so stale text did not
// reproduce in any probe against main.
it('remounts a response row when the message at its index is replaced', async () => {
  const prompt = {
    id: 'user-prompt',
    role: 'user' as const,
    content: [{ type: 'text' as const, text: 'Check both.' }],
    attachments: [],
    createdAt,
    metadata: { custom: {} }
  } as ThreadMessage

  const firstAssistant = {
    id: 'assistant-first',
    role: 'assistant' as const,
    content: [{ type: 'text' as const, text: 'Checking.' }],
    status: { type: 'complete' as const, reason: 'stop' as const },
    createdAt,
    metadata: { unstable_state: null, unstable_annotations: [], unstable_data: [], steps: [], custom: {} }
  } as ThreadMessage

  const original = {
    id: 'assistant-old',
    role: 'assistant' as const,
    content: [
      { type: 'text' as const, text: 'Old answer' },
      { type: 'text' as const, text: 'second part' }
    ],
    status: { type: 'complete' as const, reason: 'stop' as const },
    createdAt,
    metadata: { unstable_state: null, unstable_annotations: [], unstable_data: [], steps: [], custom: {} }
  } as ThreadMessage

  const replacement = {
    ...original,
    id: 'assistant-new',
    content: [{ type: 'text' as const, text: 'New answer' }]
  } as ThreadMessage

  const { container, rerender } = render(
    <ThreadRuntime messages={[prompt, firstAssistant, original]}>
      <Thread />
    </ThreadRuntime>
  )

  const originalNode = container.querySelector('[data-message-id="assistant-old"]')

  rerender(
    <ThreadRuntime messages={[prompt, firstAssistant, replacement]}>
      <Thread />
    </ThreadRuntime>
  )

  await waitFor(() => expect(container.textContent).toContain('New answer'))
  expect(container.textContent).not.toContain('Old answer')
  expect(container.textContent).not.toContain('second part')
  expect(container.querySelector('[data-message-id="assistant-new"]')).not.toBe(originalNode)
  expect(container.querySelector('[data-message-id="assistant-old"]')).toBeNull()
})

it('ends the response at a real user prompt or unrelated system event', () => {
  const messages = toChatMessages([
    { role: 'user', content: 'First question', timestamp: 1 },
    { role: 'assistant', content: 'First answer', timestamp: 2 },
    { role: 'system', content: 'slash:/model\nmodel changed', timestamp: 3 },
    { role: 'assistant', content: 'Separate answer', timestamp: 4 },
    { role: 'user', content: 'Second question', timestamp: 5 },
    { role: 'assistant', content: 'Second answer', timestamp: 6 }
  ]).map(toRuntimeMessage)

  const { container } = render(
    <ThreadRuntime messages={messages}>
      <Thread />
    </ThreadRuntime>
  )

  expect(container.querySelectorAll('[data-slot="aui_turn-pair"]')).toHaveLength(2)
  expect(container.querySelectorAll('[data-slot="aui_msg-actions"]')).toHaveLength(3)
})
