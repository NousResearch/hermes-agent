import { expect, it } from 'vitest'

import { withoutCoveredAssistantPrefix } from './coverage'
import type { ChatMessage } from './types'

const assistant = (id: string, toolId: string, result?: string): ChatMessage => ({
  id,
  role: 'assistant',
  parts: [
    { type: 'text', text: 'Repeated commentary' },
    { type: 'tool-call', toolCallId: toolId, toolName: 'read_file', args: {}, result }
  ]
})

it('retains a journaled tool result when only its invocation persisted', () => {
  const local = [assistant('live', 'call-0', 'result not yet persisted')]

  expect(withoutCoveredAssistantPrefix([assistant('stored', 'call-0')], local)).toBe(local)
})

it('consumes one proven tool occurrence without consuming equal later commentary', () => {
  const second = assistant('second', 'call-1', 'later result')
  const local = [assistant('first', 'call-0', 'first result'), second]

  expect(withoutCoveredAssistantPrefix([assistant('stored', 'call-0', 'first result')], local)).toEqual([second])
})

it('does not consume journal output across a human correction', () => {
  const correction: ChatMessage = { id: 'correction', role: 'user', parts: [{ type: 'text', text: 'Change course' }] }
  const later = assistant('later', 'call-1', 'later result')
  const local = [assistant('first', 'call-0', 'first result'), correction, later]

  expect(withoutCoveredAssistantPrefix([
    assistant('stored', 'call-0', 'first result'),
    assistant('later-stored', 'call-1', 'later result')
  ], local)).toEqual([correction, later])
})
