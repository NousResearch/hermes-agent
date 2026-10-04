import { cleanup } from '@testing-library/react'
import { afterEach, expect, it } from 'vitest'

import { chatMessageText } from '@/lib/chat-messages'

import { type GatewayFrame, playFrames } from './test-harness'

// The housekeeping-tool fallback ends a turn on the answer the model streamed
// BEFORE its tool call; the backend names that final `response_reused`. The
// merge bounds "the current response" at the bubble's last tool row, so the
// final was appended after the tool: the reply twice in one bubble, with
// nothing between the copies when the tool is silent (todo_list).

const SID = 'reused-final'
const ANSWER = 'Here is the answer.'

afterEach(cleanup)

const toolRound = (id: string, name: string): GatewayFrame[] => [
  ['tool.start', { name, tool_id: id, args: {} }],
  ['tool.complete', { name, tool_id: id, result: 'ok' }]
]

const reused = { text: ANSWER, response_previewed: true, response_reused: true }

const texts = async (frames: GatewayFrame[]) =>
  (await playFrames(SID, [['message.start', {}], ...frames])).map(message => chatMessageText(message).trim())

it.each([
  ['a silent todo_list, no interim seal', [['message.delta', { text: ANSWER }], ...toolRound('m', 'todo_list')]],
  [
    'a visible memory row, sealed interim',
    [
      ['message.delta', { text: ANSWER }],
      ['message.interim', { text: ANSWER, already_streamed: true }],
      ...toolRound('m', 'memory')
    ]
  ],
  [
    'an earlier tool round in the same bubble',
    [
      ['message.delta', { text: 'Let me look.' }],
      ...toolRound('t', 'terminal'),
      ['message.delta', { text: `\n\n${ANSWER}` }],
      ...toolRound('m', 'todo_list')
    ]
  ]
] as [string, GatewayFrame[]][])('a reused final after %s settles the reply once', async (_label, frames) => {
  const all = (await texts([...frames, ['message.complete', reused]])).join('\n')

  expect(all.split(ANSWER).length - 1).toBe(1)
})

it('a reused final whose deltas never reached this window still paints the reply', async () => {
  // Reconnect shape: the window missed the streamed answer. Settling "as is"
  // would leave the bubble without the reply.
  expect(await texts([...toolRound('m', 'todo_list'), ['message.complete', reused]])).toEqual([ANSWER])
})

it('the same words in a second response are not a reused final: both stay', async () => {
  const all = (
    await texts([
      ['message.delta', { text: 'Done.' }],
      ...toolRound('t', 'terminal'),
      ['message.delta', { text: '\n\nDone.' }],
      ['message.complete', { text: 'Done.' }]
    ])
  ).join('\n')

  expect(all.split('Done.').length - 1).toBe(2)
})
