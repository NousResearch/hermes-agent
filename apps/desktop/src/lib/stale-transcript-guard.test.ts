import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { getSessionMessages } from '@/hermes'
import { textPart } from '@/lib/chat-messages'
import { $sessions, setSessions } from '@/store/session'
import type { SessionInfo } from '@/types/hermes'

import { refreshIfTranscriptStale } from './stale-transcript-guard'

vi.mock('@/hermes', () => ({
  getSessionMessages: vi.fn(async () => ({ messages: [], session_id: 'session' }))
}))

function sessionInfo(overrides: Partial<SessionInfo> = {}): SessionInfo {
  return {
    ended_at: null,
    id: 'stored-1',
    input_tokens: 0,
    is_active: true,
    last_active: 0,
    message_count: 2,
    model: null,
    output_tokens: 0,
    preview: null,
    source: null,
    started_at: 0,
    title: 'Chat',
    tool_call_count: 0,
    ...overrides
  }
}

describe('refreshIfTranscriptStale', () => {
  beforeEach(() => {
    vi.mocked(getSessionMessages).mockReset()
    vi.mocked(getSessionMessages).mockImplementation(async () => ({ messages: [], session_id: 'stored-1' }))
    $sessions.set([])
  })

  afterEach(() => {
    $sessions.set([])
  })

  it('passes the owning profile into getSessionMessages for cross-profile routing', async () => {
    setSessions(() => [sessionInfo({ id: 'stored-1', profile: 'work-vps' })])
    vi.mocked(getSessionMessages).mockResolvedValue({
      session_id: 'stored-1',
      messages: [
        { content: 'a', role: 'user', timestamp: 1 },
        { content: 'b', role: 'assistant', timestamp: 2 },
        { content: 'c', role: 'user', timestamp: 3 }
      ]
    })

    const local = [
      { id: 'u1', role: 'user' as const, parts: [textPart('a')] },
      { id: 'a1', role: 'assistant' as const, parts: [textPart('b')] }
    ]

    const refreshed = await refreshIfTranscriptStale('stored-1', local)

    expect(getSessionMessages).toHaveBeenCalledWith('stored-1', 'work-vps')
    expect(refreshed).toHaveLength(3)
  })

  it('returns null when the remote transcript is not ahead', async () => {
    setSessions(() => [sessionInfo()])
    vi.mocked(getSessionMessages).mockResolvedValue({
      session_id: 'stored-1',
      messages: [
        { content: 'a', role: 'user', timestamp: 1 },
        { content: 'b', role: 'assistant', timestamp: 2 }
      ]
    })

    const local = [
      { id: 'u1', role: 'user' as const, parts: [textPart('a')] },
      { id: 'a1', role: 'assistant' as const, parts: [textPart('b')] }
    ]

    expect(await refreshIfTranscriptStale('stored-1', local)).toBeNull()
  })

  it('fails open (null) when the authoritative read throws', async () => {
    setSessions(() => [sessionInfo({ profile: 'missing-backend' })])
    vi.mocked(getSessionMessages).mockRejectedValue(new Error('wrong backend'))

    const local = [{ id: 'u1', role: 'user' as const, parts: [textPart('a')] }]

    expect(await refreshIfTranscriptStale('stored-1', local)).toBeNull()
  })
})
