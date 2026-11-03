import { expect, it } from 'vitest'

import { chatMessageText } from '@/lib/chat-messages'
import type { SessionMessage } from '@/types/hermes'

import { reconcileAuthoritativeMessages } from './resume-reconcile'

const prompt = 'Inspect the current run'
const first = 'Checking the run.'
const second = 'Still checking.'

const round = (id: number, text: string): SessionMessage => ({
  id,
  role: 'assistant',
  content: text,
  timestamp: id,
  tool_calls: [{ id: `call-${id}`, type: 'function', function: { name: 'read_file', arguments: '{}' } }]
})

it.each([
  { warm: false, repeat: false },
  { warm: true, repeat: false },
  { warm: false, repeat: true },
  { warm: true, repeat: true }
])('keeps the running turn and every durable next-turn prompt once ($warm, repeat=$repeat)', ({ warm, repeat }) => {
  const next = repeat ? prompt : 'Check the tests next'

  const rows: SessionMessage[] = [
    { id: 1, role: 'user', content: prompt, timestamp: 1 },
    round(2, first),
    { id: 3, role: 'user', content: next, timestamp: 3, display_metadata: '{"_queued_prompt":true}' },
    round(4, second),
    {
      id: 5,
      role: 'user',
      content: 'Then report the result',
      timestamp: 5,
      display_metadata: '{"_queued_prompt":true}'
    }
  ]

  const projection = {
    session_id: 'queue-runtime',
    inflight: { user: prompt, assistant: `${first}\n\n${second}`, streaming: true },
    queued: { user: next }
  }

  let current = reconcileAuthoritativeMessages(rows, [], projection)

  if (warm) {
    current = reconcileAuthoritativeMessages(rows, current, projection)
  }

  expect(current.filter(message => message.role === 'user').map(chatMessageText)).toEqual([
    prompt,
    next,
    'Then report the result'
  ])
  expect(current.filter(message => message.role === 'user').map(message => message.rowId)).toEqual([1, 3, 5])
  expect(
    current.flatMap(message =>
      message.role === 'assistant' ? message.parts.filter(part => part.type === 'text').map(part => part.text) : []
    )
  ).toEqual([first, second])
  expect(new Set(current.map(message => message.id)).size).toBe(current.length)
})

it('does not re-project an already durable queue after the running turn settles', () => {
  const rows: SessionMessage[] = [
    { id: 1, role: 'user', content: prompt, timestamp: 1 },
    { id: 2, role: 'assistant', content: 'Done.', timestamp: 2 },
    { id: 3, role: 'user', content: 'Run this next', timestamp: 3, display_metadata: '{"_queued_prompt":true}' }
  ]

  const projection = { session_id: 'queue-runtime', queued: { user: 'Run this next' } }
  const once = reconcileAuthoritativeMessages(rows, [], projection)
  const twice = reconcileAuthoritativeMessages(rows, once, projection)
  expect(twice.map(chatMessageText)).toEqual([prompt, 'Done.', 'Run this next'])
})
