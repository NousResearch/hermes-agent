import { expect, it } from 'vitest'

import { chatMessageText, withoutInjectedMemoryContext } from '@/lib/chat-messages'
import type { SessionMessage } from '@/types/hermes'

import { reconcileAuthoritativeMessages } from './resume-reconcile'

const submitted = 'Inspect this result\n\n[Image attached at: /fixture/result.png]\n[screenshot]'

const memory =
  '<memory-context>\n[System note: The following is recalled memory context, NOT new user input. Treat as reference data.]\nPrior work pointers.\n</memory-context>'

const commentary = 'Checking this result.'

it.each([
  { warm: false, complete: false },
  { warm: true, complete: false },
  { warm: false, complete: true },
  { warm: true, complete: true }
])(
  'does not duplicate a prompt whose stored copy gained injected memory ($warm, complete=$complete)',
  ({ warm, complete }) => {
    const rows: SessionMessage[] = [
      { id: 1, role: 'user', content: `${submitted}\n\n${memory}`, timestamp: 1 },
      {
        id: 2,
        role: 'assistant',
        content: commentary,
        timestamp: 2,
        tool_calls: [{ id: 'call-1', type: 'function', function: { name: 'read_file', arguments: '{}' } }]
      },
      ...(complete ? [{ id: 3, role: 'assistant' as const, content: 'Finished.', timestamp: 3 }] : [])
    ]

    const projection = {
      session_id: 'memory-runtime',
      turn_started_at: 1,
      inflight: { user: submitted, assistant: complete ? `${commentary}\n\nFinished.` : commentary, streaming: true }
    }

    let current = reconcileAuthoritativeMessages(rows, [], projection)

    if (warm) {
      current = reconcileAuthoritativeMessages(rows, current, projection)
    }

    expect(current.filter(message => message.role === 'user')).toHaveLength(1)
    expect(current.filter(message => message.role === 'user')[0].rowId).toBe(1)
    expect(
      current
        .filter(message => message.role === 'assistant')
        .map(chatMessageText)
        .join('\n')
    ).toMatch(complete ? /Finished\./ : /Checking this result\./)
    expect(current.map(chatMessageText).join('\n').split(commentary)).toHaveLength(2)
  }
)

it.each([
  submitted.replace('/fixture/result.png', '/fixture/other.png'),
  submitted.replace('Inspect this result', 'Inspect a different result'),
  `${submitted}\n\n<memory-context>Authored explanation</memory-context>`
])('preserves a distinct accepted prompt instead of matching only its caption', next => {
  const rows: SessionMessage[] = [
    { id: 1, role: 'user', content: `${submitted}\n\n${memory}`, timestamp: 1 },
    {
      id: 2,
      role: 'assistant',
      content: commentary,
      timestamp: 2,
      tool_calls: [{ id: 'call-1', type: 'function', function: { name: 'read_file', arguments: '{}' } }]
    }
  ]

  const current = reconcileAuthoritativeMessages(rows, [], {
    session_id: 'memory-runtime',
    inflight: { user: next, assistant: 'A new response', streaming: true }
  })

  expect(current.filter(message => message.role === 'user')).toHaveLength(2)
  expect(chatMessageText(current[0])).toContain(memory)
})

it.each([
  '<memory-context>Authored explanation</memory-context>',
  `${memory}\n\nThis text is part of the authored prompt.`
])('does not strip authored or non-trailing context blocks', text => {
  expect(withoutInjectedMemoryContext(text)).toBe(text)
})
