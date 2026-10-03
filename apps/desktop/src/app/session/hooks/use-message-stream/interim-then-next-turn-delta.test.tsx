import type { GatewayEventName } from '@hermes/shared'
import { act, cleanup } from '@testing-library/react'
import { afterEach, expect, it } from 'vitest'

import { chatMessageText } from '@/lib/chat-messages'

import { renderMessageStream } from './test-harness'

const SID = 'issue-130396-interim-then-next-turn-delta'

afterEach(cleanup)

type Frame = [GatewayEventName, Record<string, unknown>]

async function play(frames: Frame[]) {
  const stream = renderMessageStream(SID)

  for (const [type, payload] of frames) {
    await act(() => stream.handleEvent({ type, payload, session_id: SID }))
  }

  return stream.state().messages.filter(message => message.role === 'assistant' && !message.hidden)
}

// #130396, second path: the interim seals commentary that carried a tool call,
// clearing `streamId`. The next response opens a fresh bubble whose streamed
// text lands *before* its tool call, so the bubble ends on the tool round and is
// not interim at completion. The empty post-tool suffix used to make the final
// append unconditionally, painting the reply twice in that bubble.
it('does not repaint a final that already streamed before the bubble’s last tool call', async () => {
  const messages = await play([
    ['message.start', {}],
    ['message.delta', { text: 'Checking the config first.' }],
    ['tool.start', { name: 'read_file', tool_id: 's1', args: {} }],
    ['tool.complete', { name: 'read_file', tool_id: 's1', result: 'file' }],
    ['message.interim', { text: 'Checking the config first.', already_streamed: true }],

    ['message.start', {}],
    ['message.delta', { text: 'Here is the answer.' }],
    ['tool.start', { name: 'terminal', tool_id: 't1', args: {} }],
    ['tool.complete', { name: 'terminal', tool_id: 't1', result: 'out' }],
    ['message.complete', { text: 'Here is the answer.' }]
  ])

  const last = messages.at(-1)!
  expect(chatMessageText(last)).toBe('Here is the answer.')
  expect(last.parts.filter(part => part.type === 'tool-call')).toHaveLength(1)
  expect(messages.map(chatMessageText).join('\n').split('Here is the answer.').length - 1).toBe(1)
})

// Guard: a final that differs from the pre-tool text is new content and still lands.
it('still appends a genuinely new final after a bubble that ends on its tool call', async () => {
  const messages = await play([
    ['message.start', {}],
    ['message.delta', { text: 'Checking the config first.' }],
    ['tool.start', { name: 'read_file', tool_id: 's1', args: {} }],
    ['tool.complete', { name: 'read_file', tool_id: 's1', result: 'file' }],
    ['message.interim', { text: 'Checking the config first.', already_streamed: true }],

    ['message.start', {}],
    ['message.delta', { text: 'Running it now.' }],
    ['tool.start', { name: 'terminal', tool_id: 't1', args: {} }],
    ['tool.complete', { name: 'terminal', tool_id: 't1', result: 'out' }],
    ['message.complete', { text: 'All done: exit 0.' }]
  ])

  const text = messages.map(chatMessageText).join('\n')
  expect(text).toContain('Running it now.')
  expect(text).toContain('All done: exit 0.')
})
