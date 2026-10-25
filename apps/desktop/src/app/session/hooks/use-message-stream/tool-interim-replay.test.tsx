import type { GatewayEventName } from '@hermes/shared'
import { act, cleanup } from '@testing-library/react'
import { afterEach, expect, it } from 'vitest'

import { chatMessageText } from '@/lib/chat-messages'

import { renderMessageStream } from './test-harness'

afterEach(cleanup)

it.each([false, true])(
  'keeps one sealed commentary when replay lands over a tool bubble (completed: %s)',
  async completed => {
    const sid = 'tool-interim-replay'
    const stream = renderMessageStream(sid)
    const comment = 'Inspecting the fixture.'

    const send = (type: GatewayEventName, payload: Record<string, unknown> = {}) =>
      act(() => stream.handleEvent({ type, payload, session_id: sid }))

    await send('message.start')
    await send('message.delta', { text: comment })
    await send('message.interim', { text: comment, already_streamed: true })
    await send('tool.start', { name: 'read_file', tool_id: 'read-fixture', args: {} })

    if (completed) {
      await send('tool.complete', { name: 'read_file', tool_id: 'read-fixture', result: 'fixture contents' })
    }

    const liveToolStream = stream.state().streamId
    await send('message.interim', { text: comment, already_streamed: true })

    expect(
      stream
        .state()
        .messages.map(chatMessageText)
        .filter(text => text === comment)
    ).toHaveLength(1)
    expect(stream.state().streamId).toBe(liveToolStream)

    await send('tool.complete', { name: 'read_file', tool_id: 'read-fixture', result: 'fixture contents' })
    await send('message.complete', { text: 'Please provide the fixture archive.' })

    const messages = stream.state().messages
    expect(messages.map(chatMessageText).join('\n\n').split(comment)).toHaveLength(2)
    expect(messages.map(chatMessageText).join('\n\n').split('Please provide the fixture archive.')).toHaveLength(2)
    expect(messages.flatMap(message => message.parts.filter(part => part.type === 'tool-call'))).toMatchObject([
      { toolCallId: 'read-fixture', result: 'fixture contents' }
    ])
  }
)

it.each(['new text', 'new text before tool', 'new reasoning', 'new turn', 'different commentary'] as const)(
  'preserves a separate occurrence when there is %s',
  async scenario => {
    const sid = 'tool-interim-new-occurrence'
    const stream = renderMessageStream(sid)
    const comment = 'Inspecting the fixture.'

    const send = (type: GatewayEventName, payload: Record<string, unknown> = {}) =>
      act(() => stream.handleEvent({ type, payload, session_id: sid }))

    await send('message.start')
    await send('message.delta', { text: comment })
    await send('message.interim', { text: comment, already_streamed: true })
    await send('tool.start', { name: 'read_file', tool_id: 'read-fixture', args: {} })

    if (scenario === 'new text' || scenario === 'new text before tool') {
      await send('message.delta', { text: comment })

      if (scenario === 'new text before tool') {
        await send('tool.start', { name: 'read_file', tool_id: 'second-read', args: {} })
      }
    } else if (scenario === 'new reasoning') {
      await send('reasoning.delta', { text: 'A separate response is being considered.' })
    } else if (scenario === 'new turn') {
      await send('message.start')
    }

    const next = scenario === 'different commentary' ? 'Inspecting another fixture.' : comment
    await send('message.interim', { text: next, already_streamed: true })

    expect(stream.state().messages.map(chatMessageText).filter(Boolean)).toEqual([comment, next])
    expect(stream.state().streamId).toBeNull()
  }
)
