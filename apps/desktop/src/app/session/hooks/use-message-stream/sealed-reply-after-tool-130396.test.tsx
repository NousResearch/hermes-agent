import type { GatewayEventName } from '@hermes/shared'
import { act, cleanup } from '@testing-library/react'
import { afterEach, expect, it } from 'vitest'

import { chatMessageText } from '@/lib/chat-messages'

import { renderMessageStream } from './test-harness'

const SID = 'issue-130396-sealed-reply-after-tool'

afterEach(cleanup)

type Frame = [GatewayEventName, Record<string, unknown>]

const COMMENTARY = 'Confirmed: with voice off.'

const REPLY = [
  'Both lower-priority items closed out:',
  '',
  '| Item | Result |',
  '|---|---|',
  '| **STT** | fixed |',
  '| TTS | ok |'
].join('\n')

async function play(frames: Frame[]) {
  const stream = renderMessageStream(SID)

  for (const [type, payload] of frames) {
    await act(() => stream.handleEvent({ type, payload, session_id: SID }))
  }

  return stream.state().messages.filter(message => message.role === 'assistant' && !message.hidden)
}

const occurrences = (text: string, needle: string) => text.split(needle).length - 1

// #130396: commentary → tool → reply, where the reply is also sealed by a
// verify-on-stop `message.interim` before `message.complete`. The sealed bubble
// carries the earlier commentary and tool round too, so its whole text never
// matched the final; the final then painted the reply a second time as its own
// bubble. The backend stored the reply once.
it('settles the final onto the sealed bubble when the interim already holds the reply after a tool', async () => {
  const messages = await play([
    ['message.start', {}],
    ['message.delta', { text: COMMENTARY }],
    ['tool.start', { name: 'skill_manage', tool_id: 's1', args: {} }],
    ['tool.complete', { name: 'skill_manage', tool_id: 's1', result: '{"success": true}' }],
    ['message.delta', { text: REPLY }],
    ['message.interim', { text: REPLY, already_streamed: true }],
    ['message.complete', { text: REPLY }]
  ])

  expect(messages).toHaveLength(1)
  const [message] = messages
  expect(message.interim).toBe(false)
  expect(occurrences(chatMessageText(message), 'Both lower-priority items closed out')).toBe(1)
  expect(chatMessageText(message)).toContain(COMMENTARY)
  expect(message.parts.filter(part => part.type === 'tool-call')).toHaveLength(1)
})

it('still appends a genuinely different final after a sealed tool round', async () => {
  const messages = await play([
    ['message.start', {}],
    ['message.delta', { text: COMMENTARY }],
    ['tool.start', { name: 'skill_manage', tool_id: 's1', args: {} }],
    ['tool.complete', { name: 'skill_manage', tool_id: 's1', result: '{"success": true}' }],
    ['message.delta', { text: REPLY }],
    ['message.interim', { text: REPLY, already_streamed: true }],
    ['message.complete', { text: 'Something else entirely.' }]
  ])

  const text = messages.map(message => chatMessageText(message)).join('\n')

  expect(occurrences(text, 'Both lower-priority items closed out')).toBe(1)
  expect(occurrences(text, 'Something else entirely.')).toBe(1)
})
