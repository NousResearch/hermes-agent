import { describe, expect, it } from 'vitest'

import { isDraftSessionRow } from '@/app/chat/draft-session'
import type { SessionInfo } from '@/types/hermes'

const row = (overrides: Partial<SessionInfo>): SessionInfo => ({ ...({} as SessionInfo), ...overrides })

describe('isDraftSessionRow', () => {
  it('accepts a visible row with an explicit zero message count', () => {
    expect(isDraftSessionRow(row({ id: 'stored-1', message_count: 0 }))).toBe(true)
  })

  it('rejects an unknown session (no row resolves yet)', () => {
    // Reopening an old chat clears messages while its row is still loading;
    // absence of a row is not draft evidence — no splash flash mid-resume.
    expect(isDraftSessionRow(undefined)).toBe(false)
  })

  it('rejects a row without an explicit count', () => {
    // undefined counts read as unknown, not zero.
    expect(isDraftSessionRow(row({ id: 'stored-1' }))).toBe(false)
  })

  it('rejects a session with history', () => {
    expect(isDraftSessionRow(row({ id: 'stored-1', message_count: 7 }))).toBe(false)
  })

  it('rejects hidden bot plumbing rows (canonical Bot Chat, group rooms)', () => {
    // Their empty state belongs to the owning plugin's CHAT_EMPTY_AREA
    // contribution — the Hermes splash would erase the bot's face.
    expect(isDraftSessionRow(row({ id: 'stored-1', message_count: 0, hidden: true }))).toBe(false)
  })

  it('rejects a bot-scoped chat even when its visible row reads zero', () => {
    expect(isDraftSessionRow(row({ id: 'stored-1', message_count: 0 }), true)).toBe(false)
  })
})
