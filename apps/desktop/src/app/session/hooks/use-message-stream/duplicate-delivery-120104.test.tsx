import type { GatewayEventName } from '@hermes/shared'
import { act, cleanup } from '@testing-library/react'
import { afterEach, expect, it } from 'vitest'

import { chatMessageText } from '@/lib/chat-messages'

import { renderMessageStream } from './test-harness'

const SID = 'issue-120104-duplicate-delivery'

afterEach(cleanup)

type Frame = [GatewayEventName, Record<string, unknown>]

const SKILL_TURN: Frame[] = [
  ['message.start', {}],
  ['message.delta', { text: 'reply text' }],
  ['tool.start', { name: 'skill_view', tool_id: 's1', args: { name: 'x' } }],
  ['tool.complete', { name: 'skill_view', tool_id: 's1', result: 'skill content' }],
  ['message.interim', { text: 'reply text', already_streamed: true }],
  ['message.complete', { text: 'reply text' }]
]

async function mount() {
  const stream = renderMessageStream(SID)

  const send = (type: GatewayEventName, payload: Record<string, unknown> = {}) =>
    act(() => stream.handleEvent({ type, payload, session_id: SID }))

  return { stream, send }
}


it('collapses a final reply when the sealed interim contains its tail', async () => {
  const { stream, send } = await mount()
  const finalText = 'First section.\n\nSecond section.\n\nFinal section.'
  const tail = 'Second section.\n\nFinal section.'

  await send('message.start')
  await send('message.delta', { text: tail })
  await send('message.interim', { text: tail, already_streamed: true })
  await send('tool.start', { name: 'skill_view', tool_id: 's1', args: { name: 'x' } })
  await send('tool.complete', { name: 'skill_view', tool_id: 's1', result: 'skill content' })
  await send('message.complete', { text: finalText })

  const messages = stream.state().messages.filter(m => m.role === 'assistant' && !m.hidden)
  expect(messages).toHaveLength(1)
  expect(chatMessageText(messages[0])).toBe(finalText)
})

function expectSingleBubble(stream: ReturnType<typeof renderMessageStream>) {
  const messages = stream.state().messages.filter(m => m.role === 'assistant' && !m.hidden)
  expect(messages).toHaveLength(1)
  expect(chatMessageText(messages[0])).toBe('reply text')
  expect(messages[0].parts.filter(part => part.type === 'tool-call')).toHaveLength(1)
}

// #120104: an assistant reply renders twice while the backend ran a single
// turn. A redelivered message.interim lands after the first copy sealed the
// stream, so it takes the standalone-append path.
it('renders one bubble when the duplicate interim lands after the first completion', async () => {
  const { stream, send } = await mount()

  const skewed: Frame[] = [...SKILL_TURN.slice(0, 5), SKILL_TURN[5], SKILL_TURN[4], SKILL_TURN[5]]

  for (const [type, payload] of skewed) {
    await send(type, payload)
  }

  expectSingleBubble(stream)
  cleanup()
})
