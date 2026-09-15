import { describe, expect, it } from 'vitest'

import {
  $scheduledMessagesBySession,
  clearScheduledMessages,
  defaultDueInput,
  formatLocalDue,
  fromWire,
  getScheduledMessages,
  isFutureDueInput,
  parseDueInput,
  setScheduledMessages,
  toLocalInputValue
} from './scheduled-messages'

/**
 * (#111873) Scheduled composer messages — the renderer's cache of the backend's
 * list, and the one place the local wall clock is translated.
 *
 * The time helpers are the load-bearing part: `datetime-local` carries no zone,
 * so a UTC round trip would schedule messages hours off for anyone not on
 * Greenwich, and the backend stores epoch seconds, not the wall clock.
 */

describe('wire mapping', () => {
  it('converts epoch seconds to ms and drops unusable rows', () => {
    const messages = fromWire([
      { due_at: 1_800_000_000, id: 'a', status: 'pending', text: 'keep me' },
      { id: 'no-due', text: 'dropped' },
      { due_at: 1, id: '', text: 'dropped' },
      'nonsense'
    ])

    expect(messages).toEqual([
      {
        createdAt: 0,
        displayText: 'keep me',
        dueAt: 1_800_000_000_000,
        id: 'a',
        overdue: false,
        text: 'keep me'
      }
    ])
  })

  it('prefers the server-truncated preview and keeps the full text for the wire', () => {
    const [message] = fromWire([
      { created_at: 10, display_text: 'a very long…', due_at: 20, id: 'x', overdue: true, text: 'a very long draft' }
    ])

    expect(message.displayText).toBe('a very long…')
    expect(message.text).toBe('a very long draft')
    expect(message.createdAt).toBe(10_000)
    expect(message.overdue).toBe(true)
  })

  it('treats a non-list payload as empty rather than throwing', () => {
    expect(fromWire(undefined)).toEqual([])
    expect(fromWire({ messages: [] })).toEqual([])
    expect(fromWire(null)).toEqual([])
  })
})

describe('local time handling', () => {
  it('round-trips a Date through the picker value WITHOUT leaving the local zone', () => {
    const date = new Date(2026, 8, 16, 21, 30) // Sep 16 2026, 21:30 LOCAL
    const value = toLocalInputValue(date)

    expect(value).toBe('2026-09-16T21:30')
    expect(new Date(parseDueInput(value)!).getHours()).toBe(21)
    expect(new Date(parseDueInput(value)!).getMinutes()).toBe(30)
  })

  it('defaults to a whole minute in the near future', () => {
    const value = defaultDueInput(new Date(2026, 8, 16, 21, 30, 45), 5)

    expect(value).toBe('2026-09-16T21:35')
    expect(parseDueInput(value)! % 60_000).toBe(0)
  })

  it('rejects garbage and empty values instead of yielding an Invalid Date', () => {
    expect(parseDueInput('')).toBeNull()
    expect(parseDueInput('tomorrow')).toBeNull()
    expect(parseDueInput('2026-09-16')).toBeNull() // date only: no time to fire at
  })

  it('gates on a future time with a minute of slack for a slow clock', () => {
    const now = new Date(2026, 8, 16, 21, 30).getTime()

    expect(isFutureDueInput('2026-09-16T21:35', now)).toBe(true)
    expect(isFutureDueInput('2026-09-16T21:29:30', now)).toBe(true) // 30s past: slack, not a rejection
    expect(isFutureDueInput('2026-09-16T20:00', now)).toBe(false)
    expect(isFutureDueInput('nonsense', now)).toBe(false)
  })

  it('renders the due time the user picked, not UTC', () => {
    const dueAt = new Date(2026, 8, 16, 21, 30).getTime()
    const rendered = formatLocalDue(dueAt)

    expect(rendered).toContain('21:30')
    // 21:30 local is 13:30 UTC east of Greenwich; the panel must show the wall
    // clock the user picked, whatever the machine's locale renders around it.
    expect(rendered).not.toContain('13:30')
  })
})

describe('per-session cache', () => {
  it('is keyed by session so one chat never lists another chat\'s messages', () => {
    setScheduledMessages('a', [
      { createdAt: 0, displayText: 'for a', dueAt: 1, id: 'm1', overdue: false, text: 'for a' }
    ])

    expect(getScheduledMessages($scheduledMessagesBySession.get(), 'a')).toHaveLength(1)
    expect(getScheduledMessages($scheduledMessagesBySession.get(), 'b')).toEqual([])
    expect(getScheduledMessages($scheduledMessagesBySession.get(), null)).toEqual([])

    clearScheduledMessages('a')
    expect(getScheduledMessages($scheduledMessagesBySession.get(), 'a')).toEqual([])
  })
})
