// Reload parity for a steered turn: the durable rows a redirect persists
// (pre-steer assistant narration + tool rows → interrupted-checkpoint row →
// the user's correction → post-steer continuation with its own tools) must
// hydrate in that same order. A regression here re-creates "my steer shows
// way above the output" on every reload, even when the live path is correct.
//
// Row shape mirrors what the client actually receives for a real steered turn
// (state.db ids 731378-731389): row_id, reasoning, and
// tool_calls entries carrying the provider's call_id/response_item_id
// alongside id. The interrupt scaffolding lives in a server-only api_content
// sidecar the client never sees, so the checkpoint row's content is already
// clean here — that projection is pinned in chat-messages.test.ts.
import { describe, expect, it } from 'vitest'

import type { SessionMessage } from '@/types/hermes'

import { chatMessageText, toChatMessages } from './chat-messages'

const steeredTurnRows: SessionMessage[] = [
  { content: 'run the build', role: 'user', row_id: 731378, timestamp: 100 },
  {
    content: 'Kicking off the build:',
    reasoning: 'The build has to pass before docs are worth writing.',
    role: 'assistant',
    row_id: 731379,
    timestamp: 101,
    tool_calls: [
      {
        call_id: 'call-1',
        function: { arguments: '{"command":"make"}', name: 'terminal' },
        id: 'call-1',
        response_item_id: 'fc_call-1',
        type: 'function'
      }
    ]
  },
  {
    content: '{"output":"done","exit_code":0}',
    role: 'tool',
    row_id: 731380,
    timestamp: 102,
    tool_call_id: 'call-1',
    tool_name: 'terminal'
  },
  { content: 'Build is green. Now writing the docs section', role: 'assistant', row_id: 731381, timestamp: 103 },
  // Redirect checkpoint: content keeps the visible words; the provider
  // scaffolding rides api_content (never shipped to the client).
  { content: 'Now the ladder in the resolver:', role: 'assistant', row_id: 731382, timestamp: 104 },
  { content: 'actually skip docs, fix the tests instead', role: 'user', row_id: 731383, timestamp: 104.1 },
  {
    content: 'Got it — tests first.',
    role: 'assistant',
    row_id: 731384,
    timestamp: 105,
    tool_calls: [
      {
        call_id: 'call-2',
        function: { arguments: '{"command":"pytest"}', name: 'terminal' },
        id: 'call-2',
        response_item_id: 'fc_call-2',
        type: 'function'
      }
    ]
  },
  {
    content: '{"output":"3 passed","exit_code":0}',
    role: 'tool',
    row_id: 731385,
    timestamp: 106,
    tool_call_id: 'call-2',
    tool_name: 'terminal'
  },
  { content: 'Tests pass.', role: 'assistant', row_id: 731386, timestamp: 107 }
]

describe('toChatMessages on a persisted steered turn', () => {
  it('keeps the correction between the output it followed and the output it redirected', () => {
    const messages = toChatMessages(steeredTurnRows)
    const flat = messages.map(message => `${message.role}:${chatMessageText(message).slice(0, 24)}`)

    const promptIndex = messages.findIndex(m => m.role === 'user' && chatMessageText(m).includes('run the build'))
    const correctionIndex = messages.findIndex(m => m.role === 'user' && chatMessageText(m).includes('skip docs'))

    expect(promptIndex, flat.join(' | ')).toBeGreaterThanOrEqual(0)
    expect(correctionIndex, flat.join(' | ')).toBeGreaterThan(promptIndex)

    const preCorrection = messages.slice(promptIndex + 1, correctionIndex)
    const postCorrection = messages.slice(correctionIndex + 1)

    // Everything streamed before the steer renders above it…
    expect(preCorrection.some(m => chatMessageText(m).includes('Build is green'))).toBe(true)
    expect(preCorrection.some(m => m.parts.some(p => p.type === 'tool-call' && p.toolCallId === 'call-1'))).toBe(true)
    // …and nothing from after the steer leaks above it.
    expect(preCorrection.some(m => chatMessageText(m).includes('tests first'))).toBe(false)
    expect(preCorrection.some(m => m.parts.some(p => p.type === 'tool-call' && p.toolCallId === 'call-2'))).toBe(false)

    // The redirected continuation (text + its tool call + final reply) is below.
    expect(postCorrection.some(m => chatMessageText(m).includes('tests first'))).toBe(true)
    expect(postCorrection.some(m => m.parts.some(p => p.type === 'tool-call' && p.toolCallId === 'call-2'))).toBe(true)
    expect(chatMessageText(messages.at(-1)!)).toContain('Tests pass')
  })

  it('a tool-result row arriving after the correction cannot pull its call below the user bubble', () => {
    // Same turn, but the steer landed between the tool CALL row and its
    // RESULT row (accepted at the tool boundary). The result must attach to
    // the call above the correction, not materialize a new row below it.
    const rows: SessionMessage[] = [
      { content: 'go', role: 'user', timestamp: 100 },
      {
        content: '',
        role: 'assistant',
        timestamp: 101,
        tool_calls: [
          { function: { arguments: '{"command":"sleep 60"}', name: 'terminal' }, id: 'call-9', type: 'function' }
        ]
      },
      { content: 'never mind, stop that', role: 'user', timestamp: 101.5 },
      {
        content: '{"output":"interrupted"}',
        role: 'tool',
        timestamp: 102,
        tool_call_id: 'call-9',
        tool_name: 'terminal'
      },
      { content: 'Stopped.', role: 'assistant', timestamp: 103 }
    ]

    const messages = toChatMessages(rows)
    const correctionIndex = messages.findIndex(m => m.role === 'user' && chatMessageText(m).includes('never mind'))
    const callRowIndex = messages.findIndex(m => m.parts.some(p => p.type === 'tool-call' && p.toolCallId === 'call-9'))

    expect(correctionIndex).toBeGreaterThan(0)
    expect(callRowIndex).toBeGreaterThanOrEqual(0)
    expect(callRowIndex, messages.map(m => m.role).join(' | ')).toBeLessThan(correctionIndex)
    // Exactly one rendering of the call — the late result must not clone it.
    expect(
      messages.flatMap(m => m.parts).filter(p => p.type === 'tool-call' && p.toolCallId === 'call-9')
    ).toHaveLength(1)
  })
})
