import { describe, expect, it } from 'vitest'

import type { SessionInfo } from '@/types/hermes'

import { sessionRowDetails, type SessionRowFormatters } from './session-row-details'

const en: SessionRowFormatters = {
  messageCount: count => `${count} ${count === 1 ? 'message' : 'messages'}`,
  toolCallCount: count => `${count} ${count === 1 ? 'tool call' : 'tool calls'}`
}

const session = (overrides: Partial<SessionInfo> = {}): SessionInfo => ({
  ended_at: null,
  id: 's1',
  input_tokens: 0,
  is_active: false,
  last_active: 1,
  message_count: 26,
  model: 'google/gemini-3.1-pro',
  output_tokens: 0,
  preview: '  Explore\nGmail-like density tiers for session rows.  ',
  source: 'desktop',
  started_at: 1,
  title: 'Session density exploration',
  tool_call_count: 8,
  ...overrides
})

describe('session row details', () => {
  it('formats deterministic metadata without ambiguous call wording', () => {
    expect(sessionRowDetails(session({ git_branch: 'feature/menu' }), en)).toEqual({
      metadata: 'feature/menu · gemini-3.1-pro · 26 messages · 8 tool calls',
      preview: 'Explore Gmail-like density tiers for session rows.'
    })
  })

  it('uses singular labels and omits unavailable fields', () => {
    expect(
      sessionRowDetails(
        session({
          git_branch: null,
          message_count: 1,
          model: null,
          preview: null,
          title: 'Manual title',
          tool_call_count: 1
        }),
        en
      )
    ).toEqual({ metadata: '1 message · 1 tool call', preview: null })
  })

  it('omits zero counts from metadata so the sidebar stays clean', () => {
    expect(
      sessionRowDetails(session({ git_branch: null, message_count: 0, model: null, tool_call_count: 0 }), en)
    ).toEqual({ metadata: '', preview: 'Explore Gmail-like density tiers for session rows.' })
  })

  it('normalizes whitespace-only title, branch, and preview values', () => {
    expect(
      sessionRowDetails(
        session({
          git_branch: '   ',
          preview: '  ',
          title: '   '
        }),
        en
      )
    ).toEqual({ metadata: 'gemini-3.1-pro · 26 messages · 8 tool calls', preview: null })
  })

  it('omits the preview when it already supplies the displayed title', () => {
    expect(sessionRowDetails(session({ title: null }), en)).toEqual({
      metadata: 'gemini-3.1-pro · 26 messages · 8 tool calls',
      preview: null
    })
  })
})
