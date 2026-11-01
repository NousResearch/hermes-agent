import type { GatewayEventName } from '@hermes/shared'
import { act, cleanup } from '@testing-library/react'
import { afterEach, expect, it } from 'vitest'

import { chatMessageText } from '@/lib/chat-messages'

import { renderMessageStream } from './test-harness'

/**
 * #127621: an assistant response occasionally renders twice with
 * byte-identical text. The transport redelivers `message.interim`
 * (at-least-once delivery); the redelivery restates text the turn already
 * holds, so it must create no text — neither by appending a twin bubble
 * nor by sealing an unrelated live bubble with stale words.
 */

const SID = 'stale-interim-redelivery-127621'

afterEach(cleanup)

function mount() {
  const stream = renderMessageStream(SID)

  const send = (type: GatewayEventName, payload: Record<string, unknown> = {}) =>
    act(() => stream.handleEvent({ type, payload, session_id: SID }))

  return { stream, send }
}

function visibleAssistantTexts(stream: ReturnType<typeof renderMessageStream>): string[] {
  return stream
    .state()
    .messages.filter(m => m.role === 'assistant' && !m.hidden)
    .map(m => chatMessageText(m))
}

it('does not twin a sealed bubble when its interim is redelivered mid-segment', async () => {
  const { stream, send } = mount()
  await send('message.start', {})
  await send('message.interim', { text: 'pastel skies' })
  await send('message.delta', { text: 'new segment streaming' })
  // Transport replay of the first interim, arriving after later commentary
  // started streaming: must not seal the live bubble with stale words.
  await send('message.interim', { text: 'pastel skies' })

  const texts = visibleAssistantTexts(stream)
  expect(texts.filter(t => t === 'pastel skies')).toHaveLength(1)
  expect(texts).toContain('new segment streaming')
})

it('does not fold a stale interim into the live bubble on a tool turn', async () => {
  const { stream, send } = mount()
  await send('message.start', {})
  await send('message.delta', { text: 'reply text' })
  await send('tool.start', { name: 'terminal', tool_id: 't1', args: { command: 'pwd' } })
  await send('tool.complete', { name: 'terminal', tool_id: 't1', result: 'out' })
  await send('message.delta', { text: 'post-tool answer' })
  // Redelivery of the pre-tool commentary, arriving after the post-tool
  // response streamed: must not duplicate it inside the live bubble.
  await send('message.interim', { text: 'reply text' })

  const texts = visibleAssistantTexts(stream)
  expect(texts).toHaveLength(1)
  expect(texts[0]).toBe('reply textpost-tool answer')
})

it('still seals the live bubble with its own interim', async () => {
  const { stream, send } = mount()
  await send('message.start', {})
  await send('message.delta', { text: 'fresh commentary' })
  await send('message.interim', { text: 'fresh commentary' })

  const messages = stream.state().messages.filter(m => m.role === 'assistant' && !m.hidden)
  expect(messages).toHaveLength(1)
  expect(messages[0]).toMatchObject({ interim: true, pending: false })
  expect(chatMessageText(messages[0])).toBe('fresh commentary')
})

it('keeps one bubble when the interim is redelivered after the turn settled', async () => {
  const { stream, send } = mount()
  const T = 'the final answer text'
  await send('message.start', {})
  await send('message.delta', { text: T })
  await send('message.interim', { text: T })
  await send('message.complete', { text: T })
  await send('message.interim', { text: T })

  expect(visibleAssistantTexts(stream)).toEqual([T])
})
