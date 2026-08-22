import { describe, expect, it } from 'vitest'

import {
  TURN_DONE_BODY_FALLBACK,
  TURN_DONE_BODY_MAX,
  turnDoneNotificationBody
} from './turn-done-notification-body'

// Mirrors the minified Studio helper from #88488:
//   Xn(n?.content || e.title || "Message complete.", 140)
function legacyStudioBody(sessionTitle: string, assistantContent?: string): string {
  const raw = assistantContent || sessionTitle || TURN_DONE_BODY_FALLBACK
  return raw.length <= TURN_DONE_BODY_MAX ? raw : raw.slice(0, TURN_DONE_BODY_MAX)
}

describe('turnDoneNotificationBody', () => {
  it('uses the assistant reply when present', () => {
    expect(turnDoneNotificationBody('The build is green.')).toBe('The build is green.')
  })

  it('truncates long replies to 140 characters', () => {
    const reply = 'a'.repeat(200)
    expect(turnDoneNotificationBody(reply)).toBe('a'.repeat(TURN_DONE_BODY_MAX))
    expect(turnDoneNotificationBody(reply).length).toBe(TURN_DONE_BODY_MAX)
  })

  it('falls back to a generic phrase when content is missing, not the session title (#88488)', () => {
    const sessionTitle = "what's the weather in Seoul?"

    expect(legacyStudioBody(sessionTitle, undefined)).toBe(sessionTitle)
    expect(legacyStudioBody(sessionTitle, '')).toBe(sessionTitle)

    expect(turnDoneNotificationBody(undefined)).toBe(TURN_DONE_BODY_FALLBACK)
    expect(turnDoneNotificationBody('')).toBe(TURN_DONE_BODY_FALLBACK)
    expect(turnDoneNotificationBody('   ')).toBe(TURN_DONE_BODY_FALLBACK)
    expect(turnDoneNotificationBody(undefined, TURN_DONE_BODY_FALLBACK)).toBe(TURN_DONE_BODY_FALLBACK)
    expect(turnDoneNotificationBody('', TURN_DONE_BODY_FALLBACK)).not.toBe(sessionTitle)
  })

  it('ignores a blank i18n fallback and still avoids an empty body', () => {
    expect(turnDoneNotificationBody('', '')).toBe(TURN_DONE_BODY_FALLBACK)
    expect(turnDoneNotificationBody(null, '   ')).toBe(TURN_DONE_BODY_FALLBACK)
  })
})
