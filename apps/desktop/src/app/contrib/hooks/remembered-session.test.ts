import { describe, expect, it } from 'vitest'

import type { SessionInfo } from '@/types/hermes'

import { repairRememberedSession } from './remembered-session'

const session = (overrides: Partial<SessionInfo>): SessionInfo =>
  ({
    ended_at: null,
    id: 'session',
    input_tokens: 0,
    is_active: false,
    last_active: 0,
    message_count: 0,
    model: null,
    output_tokens: 0,
    preview: null,
    source: 'tui',
    started_at: 0,
    title: null,
    tool_call_count: 0,
    ...overrides
  }) as SessionInfo

describe('repairRememberedSession', () => {
  it('repairs a remembered delegate child to its parent', () => {
    expect(
      repairRememberedSession(session({ id: 'child', parent_session_id: 'parent', source: 'subagent' }))
    ).toBe('parent')
  })

  it('clears an orphaned delegate child instead of reopening it', () => {
    expect(repairRememberedSession(session({ id: 'child', source: 'subagent' }))).toBeNull()
  })

  it('keeps normal sessions', () => {
    expect(repairRememberedSession(session({ id: 'normal', source: 'tui' }))).toBe('normal')
  })

  it('keeps /branch children: parenthood is not the discriminator, source is', () => {
    expect(
      repairRememberedSession(session({ id: 'branch', parent_session_id: 'parent', source: 'tui' }))
    ).toBe('branch')
  })
})
