import { describe, expect, it } from 'vitest'

import { textPart } from './parts'
import { preserveLocalAssistantErrors } from './reconciliation'
import type { ChatMessage } from './types'

function row(id: string, role: ChatMessage['role'], text = id): ChatMessage {
  return { id, role, parts: [textPart(text, 123)], timestamp: 123 }
}

describe('memory recall across transcript hydration', () => {
  it('keeps each recall with its own matched turn, without duplicating on repeated refreshes', () => {
    const first = row('memory-recall-first', 'system', '🧠 Provider — recalled 1 memory')
    const second = row('memory-recall-second', 'system', '🧠 Provider — recalled 2 memories')

    const current = [
      row('user-one', 'user'),
      first,
      row('live-one', 'assistant', 'First answer'),
      row('user-two', 'user'),
      second,
      row('live-two', 'assistant', 'Second answer')
    ]

    const stored = [
      row('db-user-one', 'user'),
      row('db-one', 'assistant', 'First answer'),
      row('db-user-two', 'user'),
      row('db-two', 'assistant', 'Second answer')
    ]

    const hydrated = preserveLocalAssistantErrors(stored, current)
    expect(hydrated.map(message => message.id)).toEqual([
      'db-user-one',
      first.id,
      'db-one',
      'db-user-two',
      second.id,
      'db-two'
    ])
    expect(hydrated[1]).toBe(first)
    expect(hydrated[4]).toBe(second)
    expect(preserveLocalAssistantErrors(stored, hydrated)).toEqual(hydrated)
    expect(preserveLocalAssistantErrors(hydrated, hydrated)).toEqual(hydrated)
  })

  it('does not revive removed turns, cross user boundaries, or retain unrelated/hidden notices', () => {
    const current = [
      row('user-removed', 'user'),
      row('memory-recall-removed', 'system'),
      row('removed', 'assistant', 'An answer removed by compaction'),
      row('user-unanswered', 'user'),
      row('memory-recall-unanswered', 'system'),
      row('user-current', 'user'),
      row('generic-status', 'system'),
      { ...row('memory-recall-hidden', 'system'), hidden: true },
      row('live-current', 'assistant', 'Current answer')
    ]

    const stored = [row('db-user', 'user'), row('db-current', 'assistant', 'Current answer')]

    expect(preserveLocalAssistantErrors(stored, current).map(message => message.id)).toEqual(['db-user', 'db-current'])
    expect(preserveLocalAssistantErrors([], current)).toEqual([])
  })
})
