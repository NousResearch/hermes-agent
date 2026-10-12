import { expect, it } from 'vitest'

import { type ChatMessage, textPart, toChatMessages } from '@/lib/chat-messages'

import { graftRefreshedTailOntoBackfill } from './transcript-backfill'

const earlier: ChatMessage = { id: 'earlier', rowId: 1, role: 'user', parts: [textPart('Earlier')] }
const notice: ChatMessage = { id: 'notice', rowId: 5, role: 'system', parts: [textPart('Background complete')] }

const fresh = toChatMessages([
  {
    id: 2,
    role: 'assistant',
    content: '',
    reasoning: 'Inspecting',
    timestamp: 2,
    tool_calls: [{ id: 'call', type: 'function', function: { name: 'terminal', arguments: '{}' } }]
  },
  { id: 3, role: 'tool', tool_call_id: 'call', tool_name: 'terminal', content: 'ok', timestamp: 3 },
  { id: 4, role: 'assistant', content: 'Same words', timestamp: 4 },
  { id: 5, role: 'system', content: 'Background complete', timestamp: 5 }
])

it.each([
  { label: 'different durable occurrence', rowId: 6 },
  { label: 'unknown occurrence' },
  { label: 'still streaming', rowId: 4, pending: true },
  { label: 'sealed interim', rowId: 4, interim: true },
  { label: 'partial receipt', rowId: 4, durableComplete: false },
  { label: 'incomplete persisted turn', rowId: 4, persistedTurn: { row_ids: [4], complete: false } },
  { label: 'local failure', rowId: 4, error: 'Failed locally' }
])('does not discard a $label merely for equal prose', ({ label: _label, ...identity }) => {
  const local: ChatMessage = { id: 'local', role: 'assistant', parts: [textPart('Same words')], ...identity }
  const merged = graftRefreshedTailOntoBackfill(fresh, [earlier, local, notice])
  expect(merged).toContain(local)
  expect(merged).toContain(fresh[0])
})

it('keeps a cached fold when the fresh page represents only part of its sources', () => {
  const local: ChatMessage = {
    id: 'local',
    role: 'assistant',
    rowId: 4,
    parts: [
      { type: 'text', text: 'Same words', sourceRowId: 4 },
      { type: 'text', text: 'Another reply', sourceRowId: 6 }
    ]
  }

  expect(graftRefreshedTailOntoBackfill(fresh, [earlier, local, notice])).toContain(local)
})

it('keeps unstored leading rows before the replacement fold', () => {
  const prompt: ChatMessage = { id: 'user-live', role: 'user', parts: [textPart('Inspect')] }

  const local: ChatMessage = {
    id: 'local',
    role: 'assistant',
    rowId: 4,
    durableComplete: true,
    parts: [textPart('Same words')]
  }

  const merged = graftRefreshedTailOntoBackfill(fresh, [earlier, prompt, local, notice])
  expect(merged).toEqual([earlier, prompt, ...fresh])
  expect(graftRefreshedTailOntoBackfill(fresh, merged)).toEqual(merged)
})
