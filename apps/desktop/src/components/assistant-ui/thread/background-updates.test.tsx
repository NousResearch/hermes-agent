import { AssistantRuntimeProvider, useExternalStoreRuntime } from '@assistant-ui/react'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { chatMessagesEquivalent } from '@/app/session/hooks/use-session-actions/utils'
import { type ChatMessage, toChatMessages } from '@/lib/chat-messages'
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
  error = false,
  chatRows
}: {
  running?: boolean
  rows?: SessionMessage[]
  error?: boolean
  chatRows?: ChatMessage[]
}) {
  const messages = (chatRows ?? toChatMessages(rows.map((row, index) => ({ timestamp: index + 1, ...row })))).map(
    (message, index, all) =>
      toRuntimeMessage({
        ...message,
        ...(index === all.length - 1 ? { pending: running } : {}),
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
    const content = `${history[2].content}\n\n--- ✓ TASK 1/1: Review  (status=completed, api_calls=1, 1s) ---\nA **useful finding** with [evidence](https://example.com/evidence).`

    const cached = {
      id: 'cached',
      role: 'user' as const,
      parts: [{ type: 'text' as const, text: content }]
    }

    const projected = toRuntimeMessage(cached)
    expect(projected.role).toBe('system')
    expect(projected.metadata.custom?.displayKind).toBe('async_delegation_complete')
    expect(projected.metadata.custom?.asyncResult).toBe(toChatMessages([{ role: 'user', content }])[0].asyncResult)
    expect(projected.metadata.custom?.asyncResult).toContain('useful finding')
    expect(cached.role).toBe('user')
    expect(cached.parts[0].text).toBe(content)
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

  it('keeps every assistant reply outside disclosures through completion and remount', async () => {
    const view = render(<Harness running />)

    const assertRepliesVisible = () => {
      for (const row of history.filter(row => row.role === 'assistant')) {
        const reply = screen.getByText(row.content as string)
        expect(reply.closest('[data-slot="background-updates"]')).toBeNull()
      }
    }

    assertRepliesVisible()
    await act(async () => view.rerender(<Harness />))
    assertRepliesVisible()
    await waitFor(() => {
      for (const disclosure of screen.getAllByRole('button', { name: /Background updates/ })) {
        expect(disclosure.getAttribute('aria-expanded')).toBe('false')
      }
    })

    for (const disclosure of screen.getAllByRole('button', { name: /Background updates/ })) {
      expect(disclosure.getAttribute('aria-expanded')).toBe('false')
      fireEvent.click(disclosure)
      assertRepliesVisible()
      fireEvent.click(disclosure)
      assertRepliesVisible()
    }

    view.unmount()
    render(<Harness />)
    assertRepliesVisible()
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
          { role: 'user', content: 'A **useful worker result**.', display_kind: 'async_delegation_complete' },
          { role: 'assistant', content: 'The worker result is verified.' }
        ]}
      />
    )
    expect(screen.queryByText('useful worker result')).toBeNull()
    fireEvent.click(screen.getByRole('button', { name: /Background updates/ }))
    expect(screen.getByText('useful worker result')).toBeTruthy()
  })

  it('keeps unhandled results visible until a completed assistant reply exists', async () => {
    const rows: SessionMessage[] = [
      ...history.slice(0, 2),
      { role: 'user', content: 'A **result requiring attention**.', display_kind: 'async_delegation_complete' }
    ]

    const view = render(<Harness rows={rows} />)
    expect(screen.getByText('result requiring attention')).toBeTruthy()

    await act(async () => view.rerender(<Harness rows={[...rows, { role: 'user', content: 'Another request.' }]} />))
    expect(screen.getByText('result requiring attention')).toBeTruthy()

    const replied: SessionMessage[] = [...rows, { role: 'assistant', content: 'The result needs your decision.' }]
    await act(async () => view.rerender(<Harness rows={replied} running />))
    expect(screen.getByText('result requiring attention')).toBeTruthy()
    expect(screen.getByText('The result needs your decision.')).toBeTruthy()

    await act(async () => view.rerender(<Harness error rows={replied} />))
    expect(screen.getByText('result requiring attention')).toBeTruthy()

    await act(async () => view.rerender(<Harness rows={replied} />))
    expect(screen.queryByText('result requiring attention')).toBeNull()
    expect(screen.getByText('The result needs your decision.')).toBeTruthy()
  })

  it.each<SessionMessage>([
    { role: 'assistant', content: '', reasoning: 'Checking the worker result.' },
    {
      role: 'assistant',
      content: '',
      tool_calls: [{ id: 'review-check', type: 'function', function: { name: 'terminal', arguments: '{}' } }]
    }
  ])('keeps a result visible when the stopped follow-up has no reply text: %j', async followUp => {
    const rows: SessionMessage[] = [
      ...history.slice(0, 2),
      { role: 'user', content: 'UNHANDLED_REVIEW_RESULT', display_kind: 'async_delegation_complete' },
      followUp
    ]

    const view = render(<Harness rows={rows} />)
    expect(screen.getByText('UNHANDLED_REVIEW_RESULT')).toBeTruthy()

    await act(async () =>
      view.rerender(<Harness rows={[...rows, { role: 'assistant', content: 'The result needs your decision.' }]} />)
    )
    expect(screen.queryByText('UNHANDLED_REVIEW_RESULT')).toBeNull()
    expect(screen.getByText('The result needs your decision.').closest('[data-slot="background-updates"]')).toBeNull()
  })

  it('waits for a final reply after sealed interim text without crossing a new turn', async () => {
    const rows = toChatMessages([
      { role: 'user', content: 'UNHANDLED_REVIEW_RESULT', display_kind: 'async_delegation_complete', timestamp: 1 }
    ])

    // message.interim seals a ChatMessage, not a persisted SessionMessage field.
    const interim: ChatMessage = {
      id: 'interim',
      role: 'assistant',
      parts: [{ type: 'text', text: 'The result needs your decision.' }],
      interim: true,
      pending: false
    }

    const final: ChatMessage = { ...interim, id: 'final', interim: false }
    const view = render(<Harness chatRows={[...rows, interim]} />)
    expect(screen.getByText('UNHANDLED_REVIEW_RESULT')).toBeTruthy()

    for (const first of [interim, { ...interim, interim: false }]) {
      await act(async () => view.rerender(<Harness chatRows={[...rows, first, final]} running />))
      expect(screen.getByText('UNHANDLED_REVIEW_RESULT')).toBeTruthy()
      await act(async () => view.rerender(<Harness chatRows={[...rows, first, final]} error />))
      expect(screen.getByText('UNHANDLED_REVIEW_RESULT')).toBeTruthy()
    }

    for (const role of ['user', 'system'] as const) {
      const boundary: ChatMessage = { id: 'boundary', role, parts: [{ type: 'text', text: 'Another turn.' }] }
      await act(async () => view.rerender(<Harness chatRows={[...rows, interim, boundary, final]} />))
      expect(screen.getByText('UNHANDLED_REVIEW_RESULT')).toBeTruthy()
    }

    await act(async () => view.rerender(<Harness chatRows={[...rows, interim, final]} />))
    expect(screen.queryByText('UNHANDLED_REVIEW_RESULT')).toBeNull()

    for (const reply of screen.getAllByText('The result needs your decision.')) {
      expect(reply.closest('[data-slot="background-updates"]')).toBeNull()
    }
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
    expect(screen.getByText('The delayed review passed. No further changes needed.')).toBeTruthy()
  })
})
