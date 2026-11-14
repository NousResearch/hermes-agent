import { describe, expect, it } from 'vitest'

import { toChatMessages } from './hydration'

// The timeline tells injected prompts from typed ones by the row's display_kind,
// so a user row keeps it through hydration; the words a user typed carry none.
describe('toChatMessages display kind', () => {
  it('keeps display_kind on user rows only', () => {
    const messages = [
      { role: 'user', content: 'typed', timestamp: 100 },
      { role: 'user', content: '[kanban] Board updated.', timestamp: 110, display_kind: 'internal_notification' },
      { role: 'user', content: 'go left', timestamp: 120, display_kind: 'steer' },
      { role: 'assistant', content: 'ok', timestamp: 130 }
    ]

    const result = toChatMessages(messages as never)

    expect(result.map(m => [m.role, m.displayKind])).toEqual([
      ['user', undefined],
      ['user', 'internal_notification'],
      ['user', 'steer'],
      ['assistant', undefined]
    ])
  })
})
