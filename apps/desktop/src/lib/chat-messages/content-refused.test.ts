import { describe, expect, it } from 'vitest'

import type { SessionMessage } from '@/types/hermes'

import { chatMessageText, toChatMessages } from '.'

// A safety-refused turn the backend dropped from the model's context but kept
// for display. It reaches Desktop as ONE projected `user` row carrying the
// user's own message, `display_kind: 'content_refused'`, and a notice in
// `display_metadata.content_refused` (agent/turn_failure_copy.py). Hydration
// must keep it visible as a greyed, non-user-bubble row that preserves the text
// and carries the notice for the renderer.
describe('content_refused hydration', () => {
  const notice = "This message was refused by the model's safety filter — try a different query."

  const refusedRow = (
    content: string,
    displayMetadata: unknown
  ): SessionMessage => ({
    role: 'user',
    content,
    display_kind: 'content_refused',
    display_metadata: displayMetadata as SessionMessage['display_metadata'],
    timestamp: 1
  })

  it('keeps the refused turn visible as a greyed row, not a plain user bubble', () => {
    const messages = toChatMessages([refusedRow('the request the filter blocked', { content_refused: { notice } })])

    expect(messages).toHaveLength(1)
    const [message] = messages

    // (c) NOT a plain user bubble — hydrated through the greyed system path.
    expect(message.role).toBe('system')
    expect(message.role).not.toBe('user')
    // (a) the user's own words survive for display.
    expect(chatMessageText(message)).toBe('the request the filter blocked')
    // (b) the notice rides along for the renderer.
    expect(message.contentRefused).toEqual({ notice })
  })

  it('still marks the row refused when an older backend sent no notice', () => {
    const [message] = toChatMessages([refusedRow('blocked text', { content_refused: {} })])

    expect(message.role).toBe('system')
    expect(chatMessageText(message)).toBe('blocked text')
    // The field is present (so the renderer applies the greyed treatment) but
    // empty — the component fills an i18n fallback when there is no notice.
    expect(message.contentRefused).toEqual({})
  })

  it('reads a notice from display_metadata served as raw JSON text (older backend)', () => {
    const [message] = toChatMessages([
      refusedRow('blocked text', JSON.stringify({ content_refused: { notice } }))
    ])

    expect(message.role).toBe('system')
    expect(message.contentRefused).toEqual({ notice })
  })
})
