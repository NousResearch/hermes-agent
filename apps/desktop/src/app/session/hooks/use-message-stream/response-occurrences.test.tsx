import type { GatewayEventName } from '@hermes/shared'
import { act, cleanup } from '@testing-library/react'
import { afterEach, expect, it } from 'vitest'

import { type ChatMessage, chatMessageText, toChatMessages } from '@/lib/chat-messages'
import type { SessionMessage } from '@/types/hermes'

import { renderMessageStream } from './test-harness'

const text = (messages: ChatMessage[]) => messages.map(chatMessageText).join('').replace(/\s+/g, '')

afterEach(cleanup)

it.each(['interim', 'completion', 'completion without delta'])(
  'retains earlier response occurrences in a shared tool bubble on %s',
  async boundary => {
    const sid = 'response-occurrences'
    const stream = renderMessageStream(sid)

    const send = (type: GatewayEventName, payload: Record<string, unknown> = {}) =>
      act(() => stream.handleEvent({ type, payload, session_id: sid }))

    const history: SessionMessage[] = []
    const comments = boundary === 'interim' ? ['Checking.', 'Checking.', 'Next phase.'] : ['Checking.', 'Checking.']

    await send('message.start')

    for (const [index, comment] of comments.entries()) {
      await send('message.delta', { text: `${index ? '\n\n' : ''}${comment}` })

      // The actual producer suppresses the second equal interim, not its delta.
      if (index !== 1) {
        await send('message.interim', { text: comment, already_streamed: true })
      }

      const id = `tool-${index}`
      await send('tool.start', { name: 'read_file', tool_id: id, args: {} })
      await send('tool.complete', { name: 'read_file', tool_id: id, result: 'read' })
      history.push(
        {
          id: index * 2 + 1,
          role: 'assistant',
          content: comment,
          tool_calls: [{ id, type: 'function', function: { name: 'read_file', arguments: '{}' } }]
        },
        { id: index * 2 + 2, role: 'tool', content: 'read', tool_call_id: id }
      )
    }

    if (boundary !== 'completion without delta') {
      await send('message.delta', { text: '\n\nDone.' })
    }

    await send('message.complete', { text: 'Done.' })
    history.push({ id: 20, role: 'assistant', content: 'Done.' })
    const durable = toChatMessages(history)
    expect(text(stream.state().messages)).toBe(text(durable))

    const toolIds = () =>
      stream
        .state()
        .messages.flatMap(row => row.parts.flatMap(part => (part.type === 'tool-call' ? [part.toolCallId] : [])))

    expect(toolIds()).toEqual(comments.map((_, index) => `tool-${index}`))

    await send('message.complete', { text: 'Done.' })
    expect(text(stream.state().messages)).toBe(text(durable))
    expect(toolIds()).toEqual(comments.map((_, index) => `tool-${index}`))
  }
)
