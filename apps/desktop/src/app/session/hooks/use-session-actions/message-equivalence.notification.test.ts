import { describe, expect, it } from 'vitest'

import { toChatMessages } from '@/lib/chat-messages'

import { chatMessagesEquivalent } from './message-equivalence'

describe('internal-notification resume provenance', () => {
  it('publishes a changed notification turn boundary even when its visible text is unchanged', () => {
    const [turn] = toChatMessages([
      { role: 'user', content: 'Notice', display_kind: 'internal_notification', timestamp: 1 }
    ])

    const [note] = toChatMessages([
      {
        role: 'user',
        content: 'Notice',
        display_kind: 'internal_notification',
        display_metadata: { completion_admission: 'note-id' },
        timestamp: 1
      }
    ])

    expect(chatMessagesEquivalent(turn, note)).toBe(false)
    expect(chatMessagesEquivalent(turn, { ...turn })).toBe(true)
  })
})
