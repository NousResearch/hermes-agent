import { describe, expect, it } from 'vitest'

import type { SessionInfo } from '@/hermes'

import { rememberedSessionIdForRoute } from './remembered-session'

function session(overrides: Partial<SessionInfo> & Pick<SessionInfo, 'id'>): SessionInfo {
  return {
    ended_at: null,
    input_tokens: 0,
    is_active: false,
    last_active: 0,
    message_count: 0,
    model: null,
    output_tokens: 0,
    preview: null,
    source: null,
    started_at: 0,
    title: null,
    tool_call_count: 0,
    ...overrides
  }
}

describe('rememberedSessionIdForRoute', () => {
  it('remembers ordinary routed sessions unchanged', () => {
    expect(rememberedSessionIdForRoute('parent', [session({ id: 'parent', source: 'desktop' })])).toBe('parent')
  })

  it('maps delegate subagent routes back to the parent session', () => {
    const sessions = [session({ id: 'child', source: 'subagent', _delegate_from: 'parent' })]

    expect(rememberedSessionIdForRoute('child', sessions)).toBe('parent')
  })

  it('does not persist an orphan subagent as the remembered chat', () => {
    const sessions = [session({ id: 'child', source: 'subagent' })]

    expect(rememberedSessionIdForRoute('child', sessions)).toBeNull()
  })

  it('keeps unknown route ids so cold-start restore still works before the list loads', () => {
    expect(rememberedSessionIdForRoute('late-session', [])).toBe('late-session')
  })
})
