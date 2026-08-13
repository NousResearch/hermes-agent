import { afterEach, describe, expect, it, vi } from 'vitest'

import { getSession } from '@/hermes'
import { $sessions } from '@/store/session'
import type { SessionInfo } from '@/types/hermes'

import { __resetSessionLinkTitleCache, fetchSessionLinkTitle, lookupLocalSessionTitle } from './session-link-title'
import { sessionRefCacheKey } from './session-refs'

vi.mock('@/hermes', () => ({
  getSession: vi.fn()
}))

function makeSession(overrides: Partial<SessionInfo> = {}): SessionInfo {
  return {
    ended_at: null,
    id: '20260101_abc123',
    input_tokens: 0,
    is_active: false,
    last_active: 1_000,
    message_count: 1,
    model: null,
    output_tokens: 0,
    preview: null,
    profile: 'default',
    source: 'cli',
    started_at: 1_000,
    title: 'Research notes',
    tool_call_count: 0,
    ...overrides
  }
}

afterEach(() => {
  __resetSessionLinkTitleCache()
  $sessions.set([])
  vi.mocked(getSession).mockReset()
})

describe('lookupLocalSessionTitle', () => {
  it('reads from the in-memory session list', () => {
    $sessions.set([makeSession({ profile: 'work', title: 'Branch plan' })])

    expect(lookupLocalSessionTitle('work/20260101_abc123')).toBe('Branch plan')
  })

  it('matches the lineage root so a compressed chat still resolves', () => {
    $sessions.set([makeSession({ _lineage_root_id: '20260101_abc123', id: '20260102_tip', title: 'Compressed chat' })])

    expect(lookupLocalSessionTitle('20260101_abc123')).toBe('Compressed chat')
  })

  it('ignores a same-id row owned by another profile', () => {
    $sessions.set([makeSession({ profile: 'work', title: 'Work chat' })])

    expect(lookupLocalSessionTitle('personal/20260101_abc123')).toBe('')
  })

  it('returns empty for an untitled row so the caller can fall back to the id', () => {
    $sessions.set([makeSession({ preview: null, title: null })])

    expect(lookupLocalSessionTitle('default/20260101_abc123')).toBe('')
  })
})

describe('fetchSessionLinkTitle', () => {
  it('dedupes concurrent lookups', async () => {
    vi.mocked(getSession).mockResolvedValue(makeSession({ title: 'From API' }))

    const value = 'default/20260101_abc123'
    const [first, second] = await Promise.all([fetchSessionLinkTitle(value), fetchSessionLinkTitle(value)])

    expect(first).toBe('From API')
    expect(second).toBe('From API')
    expect(getSession).toHaveBeenCalledTimes(1)
    expect(getSession).toHaveBeenCalledWith('20260101_abc123', 'default')
  })

  it('uses the local sidebar row before calling the API', async () => {
    $sessions.set([makeSession({ title: 'Cached title' })])

    await expect(fetchSessionLinkTitle('default/20260101_abc123')).resolves.toBe('Cached title')
    expect(getSession).not.toHaveBeenCalled()
  })

  it('keeps separate cache entries per profile', async () => {
    vi.mocked(getSession).mockImplementation(async (id, profile) =>
      makeSession({ id, profile: profile ?? 'default', title: profile === 'work' ? 'Work chat' : 'Home chat' })
    )

    await expect(fetchSessionLinkTitle('default/20260101_abc123')).resolves.toBe('Home chat')
    await expect(fetchSessionLinkTitle('work/20260101_abc123')).resolves.toBe('Work chat')
    expect(sessionRefCacheKey('default/20260101_abc123')).not.toBe(sessionRefCacheKey('work/20260101_abc123'))
  })

  it('falls back to the preview when the session has no title', async () => {
    vi.mocked(getSession).mockResolvedValue(makeSession({ preview: 'Summarize this repo', title: null }))

    await expect(fetchSessionLinkTitle('20260101_abc123')).resolves.toBe('Summarize this repo')
  })

  it('resolves empty when the id is not on this backend', async () => {
    vi.mocked(getSession).mockRejectedValue(new Error('Session not found'))

    await expect(fetchSessionLinkTitle('default/missing')).resolves.toBe('')
  })

  it('resolves empty when the desktop bridge is unavailable', async () => {
    vi.mocked(getSession).mockImplementation(() => {
      throw new TypeError("Cannot read properties of undefined (reading 'api')")
    })

    await expect(fetchSessionLinkTitle('default/20260101_abc123')).resolves.toBe('')
  })
})
