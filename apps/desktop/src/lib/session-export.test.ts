// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { SessionInfo } from '@/types/hermes'

const getSessionMessages = vi.fn(async (_id: string, _profile?: null | string) => ({ messages: [] }))

vi.mock('@/hermes', () => ({
  getSessionMessages: (id: string, profile?: null | string) => getSessionMessages(id, profile)
}))

vi.mock('@/i18n', () => ({
  translateNow: (key: string) => key
}))

vi.mock('@/store/notifications', () => ({
  notify: vi.fn(),
  notifyError: vi.fn()
}))

const { exportSession } = await import('./session-export')

function makeSession(overrides: Partial<SessionInfo> = {}): SessionInfo {
  return {
    archived: false,
    cwd: '/home/user/projects/hermes-agent',
    ended_at: null,
    id: '20260603_090200_abcd12',
    input_tokens: 0,
    is_active: false,
    last_active: 1_000,
    message_count: 2,
    model: 'claude',
    output_tokens: 0,
    preview: 'Export session transcript',
    source: 'cli',
    started_at: 1_000,
    title: 'Remote profile session',
    tool_call_count: 0,
    ...overrides
  }
}

beforeEach(() => {
  getSessionMessages.mockClear()
  // jsdom doesn't implement object URLs; the export builds a download blob.
  URL.createObjectURL = vi.fn(() => 'blob:mock')
  URL.revokeObjectURL = vi.fn()
})

afterEach(() => {
  vi.restoreAllMocks()
})

describe('exportSession', () => {
  it('reads the transcript from the session-owning profile', async () => {
    const session = makeSession({ profile: 'work' })

    await exportSession(session.id, { session, title: session.title })

    expect(getSessionMessages).toHaveBeenCalledWith(session.id, 'work')
  })

  it('honors an explicit profile over the session profile', async () => {
    const session = makeSession({ profile: 'work' })

    await exportSession(session.id, { profile: 'research', session })

    expect(getSessionMessages).toHaveBeenCalledWith(session.id, 'research')
  })

  it('falls back to the current profile when none is known', async () => {
    await exportSession('20260603_090200_abcd12', { title: 'Local session' })

    expect(getSessionMessages).toHaveBeenCalledWith('20260603_090200_abcd12', null)
  })
})
