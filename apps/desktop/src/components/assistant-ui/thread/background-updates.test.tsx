import { AssistantRuntimeProvider, useExternalStoreRuntime } from '@assistant-ui/react'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { chatMessagesEquivalent } from '@/app/session/hooks/use-session-actions/utils'
import { toChatMessages } from '@/lib/chat-messages'
import { toRuntimeMessage } from '@/lib/chat-runtime'
import type { SessionMessage } from '@/types/hermes'

import { stubThreadEnvironment } from '../test-utils'

import { Thread } from '.'

stubThreadEnvironment()
afterEach(cleanup)

const history: SessionMessage[] = [
  { role: 'user', content: 'Fix the sidebar.' },
  { role: 'assistant', content: 'The fix is ready. Tests passed.' },
  {
    role: 'user',
    content:
      '[ASYNC DELEGATION BATCH COMPLETE — deleg_review]\nA background fan-out unit you dispatched earlier — 1 subagent(s) — has finished; its consolidated results are below.'
  },
  { role: 'assistant', content: 'The delayed review passed. No further changes needed.' },
  { role: 'user', content: 'opaque internal payload', display_kind: 'async_delegation_complete' },
  { role: 'assistant', content: 'The second review found a follow-up worth checking.' }
]

function Harness({
  running = false,
  rows = history,
  error = false
}: {
  running?: boolean
  rows?: SessionMessage[]
  error?: boolean
}) {
  const messages = toChatMessages(rows).map((message, index, all) =>
    toRuntimeMessage({
      ...message,
      ...(index === all.length - 1 && error ? { error: 'Review failed' } : {})
    })
  )

  const runtime = useExternalStoreRuntime({ messages, isRunning: running, onNew: async () => {} })

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <Thread />
    </AssistantRuntimeProvider>
  )
}

describe('background handoffs in the transcript', () => {
  it('repaints when history adds the background boundary type', () => {
    const message = toChatMessages([{ role: 'system', content: 'background agent work finished' }])[0]
    expect(chatMessagesEquivalent(message, { ...message, displayKind: 'async_delegation_complete' })).toBe(false)
  })

  it('repairs old warm-cache messages without changing the cached data', () => {
    const cached = {
      id: 'cached',
      role: 'user' as const,
      parts: [{ type: 'text' as const, text: history[2].content as string }]
    }

    const projected = toRuntimeMessage(cached)
    expect(projected.role).toBe('system')
    expect(projected.metadata.custom?.displayKind).toBe('async_delegation_complete')
    expect(cached.role).toBe('user')
  })

  it('leaves ordinary mentions and quoted protocol examples alone', () => {
    for (const content of [
      'Explain ASYNC DELEGATION BATCH COMPLETE.',
      `Example:\n${history[2].content}`,
      '```\n' + history[2].content + '\n```'
    ]) {
      expect(toChatMessages([{ role: 'user', content }])[0].role).toBe('user')
    }

    expect(toChatMessages([{ role: 'assistant', content: history[2].content }])[0].role).toBe('assistant')
  })

  it('keeps the main answer visible and discloses completed follow-ups without fake user bubbles', () => {
    render(<Harness />)
    expect(screen.getByText('The fix is ready. Tests passed.')).toBeTruthy()
    expect(screen.queryByText(/ASYNC DELEGATION/)).toBeNull()
    expect(screen.queryByText('The delayed review passed. No further changes needed.')).toBeNull()
    const disclosure = screen.getByRole('button', { name: /Background updates/ })
    expect(disclosure.getAttribute('aria-expanded')).toBe('false')
    fireEvent.click(disclosure)
    expect(screen.getByText('The delayed review passed. No further changes needed.')).toBeTruthy()
    expect(screen.getByText('The second review found a follow-up worth checking.')).toBeTruthy()
    expect(screen.queryByText(/ASYNC DELEGATION/)).toBeNull()
  })

  it('never folds ongoing work', () => {
    render(<Harness running />)
    expect(screen.getByText('The second review found a follow-up worth checking.')).toBeTruthy()
  })

  it('preserves the upstream Markdown result body inside the disclosure', () => {
    render(
      <Harness
        rows={[
          ...history.slice(0, 2),
          { role: 'user', content: 'A **useful worker result**.', display_kind: 'async_delegation_complete' }
        ]}
      />
    )
    expect(screen.queryByText('useful worker result')).toBeNull()
    fireEvent.click(screen.getByRole('button', { name: /Background updates/ }))
    expect(screen.getByText('useful worker result')).toBeTruthy()
  })

  it('never folds a failed follow-up', () => {
    render(<Harness error />)
    expect(screen.getByText('The second review found a follow-up worth checking.')).toBeTruthy()
  })

  it('does not absorb the next real user turn', () => {
    render(
      <Harness
        rows={[
          ...history,
          { role: 'user', content: 'Now explain the change.' },
          { role: 'assistant', content: 'Here is the explanation.' }
        ]}
      />
    )
    expect(screen.getByText('Now explain the change.')).toBeTruthy()
    expect(screen.getByText('Here is the explanation.')).toBeTruthy()
    expect(screen.queryByText('The delayed review passed. No further changes needed.')).toBeNull()
  })
})
