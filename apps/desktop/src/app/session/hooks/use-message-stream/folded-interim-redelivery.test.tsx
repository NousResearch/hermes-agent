import type { GatewayEventName } from '@hermes/shared'
import { act, cleanup } from '@testing-library/react'
import { afterEach, expect, it } from 'vitest'

import type { ClientSessionState } from '@/app/types'
import { type ChatMessage, chatMessageText, toChatMessages } from '@/lib/chat-messages'
import { createClientSessionState } from '@/lib/chat-runtime'
import type { SessionMessage } from '@/types/hermes'

import { renderMessageStream } from './test-harness'

afterEach(cleanup)

it.each([1, 9])('does not append replayed commentary already carried by a %i-round hydrated tool fold', async count => {
  const sid = 'folded-interim-replay'
  const comments = Array.from({ length: count }, (_, index) => `Inspecting fixture ${index + 1}.`)
  const rows: SessionMessage[] = [{ id: 1, role: 'user', content: 'Inspect the fixtures' }]

  for (const [index, comment] of comments.entries()) {
    const toolId = `read-${index}`
    rows.push(
      {
        id: index * 2 + 2,
        role: 'assistant',
        content: comment,
        tool_calls: [{ id: toolId, type: 'function', function: { name: 'read_file', arguments: '{}' } }]
      },
      { id: index * 2 + 3, role: 'tool', content: 'fixture', tool_call_id: toolId }
    )
  }

  const hydrated = toChatMessages(rows)

  const states = new Map<string, ClientSessionState>([
    [
      sid,
      {
        ...createClientSessionState(),
        messages: hydrated,
        busy: true,
        turnLive: true,
        streamId: count === 9 ? 'orphaned-stream' : null
      }
    ]
  ])

  const stream = renderMessageStream(sid, { states })

  const send = (type: GatewayEventName, payload: Record<string, unknown>) =>
    act(() => stream.handleEvent({ type, payload, session_id: sid }))

  for (const text of comments) {
    await send('message.interim', { text, already_streamed: true })
  }

  const afterReplay = stream.state().messages as ChatMessage[]
  expect(afterReplay).toEqual(hydrated)
  expect(stream.state().streamId).toBeNull()

  await send('message.delta', { text: 'A newly observed result.' })
  await send('message.interim', { text: 'A newly observed result.', already_streamed: true })
  await send('message.complete', { text: 'Finished inspecting fixtures.' })

  const text = stream.state().messages.map(chatMessageText).join('\n\n')

  for (const comment of [...comments, 'A newly observed result.', 'Finished inspecting fixtures.']) {
    expect(text.split(comment)).toHaveLength(2)
  }

  expect(
    stream.state().messages.flatMap(message => message.parts.filter(part => part.type === 'tool-call'))
  ).toHaveLength(count)
})

it.each(['new prompt', 'live response', 'hidden history', 'shared prefix', 'completed turn'] as const)(
  'keeps new commentary with a %s instead of consuming another occurrence',
  async scenario => {
    const sid = 'folded-interim-boundaries'
    const comment = 'Inspecting a fixture.'

    const rows: SessionMessage[] = [
      { id: 1, role: 'user', content: 'Inspect the fixtures' },
      {
        id: 2,
        role: 'assistant',
        content: comment,
        tool_calls: [{ id: 'read-1', type: 'function', function: { name: 'read_file', arguments: '{}' } }]
      },
      { id: 3, role: 'tool', content: 'fixture', tool_call_id: 'read-1' }
    ]

    if (scenario === 'completed turn') {
      rows.push({ id: 4, role: 'assistant', content: 'Previous final answer.' })
    }

    const hydrated = toChatMessages(rows)

    if (scenario === 'new prompt') {
      hydrated.push({ id: 'user-next', role: 'user', parts: [{ type: 'text', text: 'Inspect it again' }] })
    } else if (scenario === 'hidden history') {
      hydrated[1] = { ...hydrated[1], hidden: true }
    }

    const states = new Map<string, ClientSessionState>([
      [sid, { ...createClientSessionState(), messages: hydrated, busy: true, turnLive: true }]
    ])

    const stream = renderMessageStream(sid, { states })

    const send = (type: GatewayEventName, payload: Record<string, unknown>) =>
      act(() => stream.handleEvent({ type, payload, session_id: sid }))

    const text = scenario === 'shared prefix' ? `${comment} More detail.` : comment

    if (scenario === 'completed turn') {
      await send('message.start', {})
    }

    if (scenario === 'live response') {
      await send('message.delta', { text })
    }

    await send('message.interim', { text, already_streamed: true })

    expect(stream.state().messages).toHaveLength(hydrated.length + 1)
    expect(chatMessageText(stream.state().messages.at(-1)!)).toBe(text)
    expect(stream.state().messages.slice(0, hydrated.length)).toEqual(hydrated)
  }
)
