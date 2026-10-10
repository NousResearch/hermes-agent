import type { GatewayEventName } from '@hermes/shared'
import { act, cleanup, render, waitFor } from '@testing-library/react'
import { afterEach, expect, it } from 'vitest'

import { graftRefreshedTailOntoBackfill } from '@/app/chat/transcript-backfill'
import { stubThreadEnvironment, stubThreadViewportSize, ThreadRuntime } from '@/components/assistant-ui/test-utils'
import { Thread } from '@/components/assistant-ui/thread'
import { chatMessageText, toChatMessages } from '@/lib/chat-messages'
import { toRuntimeMessage } from '@/lib/chat-runtime'
import type { SessionMessage } from '@/types/hermes'

import { preserveLocalPendingTurnMessages } from '../use-session-actions/utils'

import { renderMessageStream } from './test-harness'

const SID = 'background-reply'
afterEach(cleanup)

it('refreshes folded background replies without keeping their settled live copies', async () => {
  const h = renderMessageStream(SID)

  const send = (type: GatewayEventName, payload: Record<string, unknown> = {}) =>
    act(() => h.handleEvent({ type, payload, session_id: SID }))

  const rows: SessionMessage[] = [
    { id: 90, role: 'user', content: 'Earlier prompt.', timestamp: 90 },
    { id: 91, role: 'assistant', content: 'Earlier reply.', timestamp: 91 }
  ]

  h.states.set(SID, { ...h.state(), messages: toChatMessages(rows) })

  for (const [index, text] of ['First response.', 'Background response.'].entries()) {
    const id = 100 + index * 10

    const prompt: SessionMessage = {
      id,
      role: 'user',
      content: index ? '[ASYNC DELEGATION BATCH COMPLETE — example]' : 'Inspect this.',
      timestamp: id,
      ...(index ? { display_kind: 'async_delegation_complete' } : {})
    }

    rows.push(prompt)
    h.states.set(SID, { ...h.state(), messages: [...h.state().messages, ...toChatMessages([prompt])] })
    await send('message.start')
    await send('reasoning.delta', { text: 'Inspecting.' })
    await send('tool.start', { name: 'terminal', tool_id: `call-${index}`, args: { command: 'pwd' } })
    await send('tool.complete', { name: 'terminal', tool_id: `call-${index}`, result: 'done' })
    await send('message.delta', { text })
    await send('message.complete', {
      text,
      persisted_turn: {
        row_ids: [id, id + 1, id + 2, id + 3],
        user_row_id: id,
        final_assistant_row_id: id + 3,
        complete: true
      }
    })
    rows.push(
      {
        id: id + 1,
        role: 'assistant',
        content: '',
        reasoning: 'Inspecting.',
        timestamp: id + 1,
        tool_calls: [
          { id: `call-${index}`, type: 'function', function: { name: 'terminal', arguments: '{"command":"pwd"}' } }
        ]
      },
      {
        id: id + 2,
        role: 'tool',
        content: 'done',
        tool_call_id: `call-${index}`,
        tool_name: 'terminal',
        timestamp: id + 2
      },
      { id: id + 3, role: 'assistant', content: text, timestamp: id + 3 }
    )
  }

  // A newest page starts inside the first turn. Its first folded row is not
  // the live final's rowId; the background notice is the later shared anchor.
  const fresh = toChatMessages(rows.slice(3))
  let previous = h.state().messages

  for (let refresh = 0; refresh < 2; refresh++) {
    previous = preserveLocalPendingTurnMessages(graftRefreshedTailOntoBackfill(fresh, previous), previous)
    expect(previous.filter(m => m.role === 'assistant').map(chatMessageText)).toEqual([
      'Earlier reply.',
      'First response.',
      'Background response.'
    ])
    expect(
      previous
        .flatMap(m => m.parts)
        .filter(p => p.type === 'tool-call')
        .map(p => p.toolCallId)
    ).toEqual(['call-0', 'call-1'])
    expect(previous.filter(m => m.role === 'user').map(chatMessageText)).toEqual(['Earlier prompt.', 'Inspect this.'])
  }

  stubThreadEnvironment()
  stubThreadViewportSize()

  const view = render(
    <ThreadRuntime messages={previous.map(toRuntimeMessage)}>
      <Thread />
    </ThreadRuntime>
  )

  await waitFor(() => {
    expect(view.container.textContent?.split('First response.')).toHaveLength(2)
    expect(view.container.textContent?.split('Background response.')).toHaveLength(2)
  })
  view.rerender(
    <ThreadRuntime messages={toChatMessages(rows).map(toRuntimeMessage)}>
      <Thread />
    </ThreadRuntime>
  )
  await waitFor(() => {
    expect(view.container.textContent?.split('First response.')).toHaveLength(2)
    expect(view.container.textContent?.split('Background response.')).toHaveLength(2)
  })
})
