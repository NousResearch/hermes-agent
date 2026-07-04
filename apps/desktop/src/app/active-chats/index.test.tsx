import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { getSessionMessages } from '@/hermes'
import {
  $attentionSessionIds,
  $messagingSessions,
  $sessions,
  $sessionsLoading,
  $workingSessionIds
} from '@/store/session'
import type { SessionInfo, SessionMessage } from '@/types/hermes'

import { ActiveChatsView } from './index'

vi.mock('@/hermes', () => ({
  getSessionMessages: vi.fn()
}))

vi.mock('@/store/notifications', () => ({
  notify: vi.fn(),
  notifyError: vi.fn()
}))

function sessionInfo(overrides: Partial<SessionInfo> = {}): SessionInfo {
  return {
    ended_at: null,
    id: 'session-1',
    input_tokens: 0,
    is_active: true,
    last_active: 1000,
    message_count: 1,
    model: null,
    output_tokens: 0,
    preview: null,
    source: null,
    started_at: 1000,
    title: 'Session',
    tool_call_count: 0,
    ...overrides
  }
}

function message(role: SessionMessage['role'], content: string): SessionMessage {
  return { content, role, timestamp: 1 }
}

describe('ActiveChatsView', () => {
  beforeEach(() => {
    vi.setSystemTime(new Date('2026-07-04T12:00:00.000Z'))
    $sessionsLoading.set(false)
    $messagingSessions.set([])
    $workingSessionIds.set(['running-chat'])
    $attentionSessionIds.set(['waiting-chat'])
    $sessions.set([
      sessionInfo({
        id: 'recent-chat',
        last_active: Date.now() / 1000 - 60,
        preview: 'Lower chat preview',
        title: 'Lower Hermes chat'
      }),
      sessionInfo({
        id: 'waiting-chat',
        last_active: Date.now() / 1000 - 120,
        preview: 'Top chat preview',
        profile: 'endlessblink',
        title: 'Top active chat'
      }),
      sessionInfo({
        id: 'running-chat',
        last_active: Date.now() / 1000 - 180,
        preview: 'Running chat preview',
        title: 'Running chat'
      })
    ])
    vi.mocked(getSessionMessages).mockResolvedValue({
      messages: [message('user', 'Need help here'), message('assistant', 'I can help')],
      session_id: 'waiting-chat'
    })
  })

  afterEach(() => {
    cleanup()
    vi.useRealTimers()
    vi.restoreAllMocks()
    $sessions.set([])
    $messagingSessions.set([])
    $workingSessionIds.set([])
    $attentionSessionIds.set([])
    $sessionsLoading.set(true)
  })

  it('shows only active chats, groups them, selects the highest-priority chat, and replies to that session', async () => {
    const onSendReply = vi.fn(async () => true)

    render(
      <ActiveChatsView
        onOpenSession={vi.fn()}
        onRefreshSessions={vi.fn(async () => undefined)}
        onSendReply={onSendReply}
      />
    )

    const rows = screen.getAllByRole('button', { name: /chat/i })

    expect(rows[0]?.textContent).toContain('Top active chat')
    expect(rows[1]?.textContent).toContain('Running chat')
    expect(screen.queryByText('Lower Hermes chat')).toBeNull()
    expect(screen.getByRole('button', { name: /Status/i })).toBeTruthy()
    expect(screen.getByRole('button', { name: /Profile/i })).toBeTruthy()
    expect(screen.getByRole('button', { name: /Workspace/i })).toBeTruthy()

    await waitFor(() =>
      expect(getSessionMessages).toHaveBeenCalledWith('waiting-chat', 'endlessblink')
    )
    expect(await screen.findByText('Need help here')).toBeTruthy()

    const replyBox = screen.getByPlaceholderText('Reply to this chat...')
    fireEvent.change(replyBox, { target: { value: 'Answer the top one' } })
    fireEvent.click(screen.getByRole('button', { name: /Answer/i }))

    await waitFor(() => expect(onSendReply).toHaveBeenCalledTimes(1))
    expect(onSendReply).toHaveBeenCalledWith(expect.objectContaining({ id: 'waiting-chat' }), 'Answer the top one')
  })

  it('keeps row selection tied to the clicked active stored session', async () => {
    vi.mocked(getSessionMessages).mockImplementation(async (sessionId: string) => ({
      messages: [message('user', `Transcript for ${sessionId}`)],
      session_id: sessionId
    }))

    render(
      <ActiveChatsView
        onOpenSession={vi.fn()}
        onRefreshSessions={vi.fn(async () => undefined)}
        onSendReply={vi.fn(async () => true)}
      />
    )

    fireEvent.click(screen.getByRole('button', { name: /Running chat/i }))

    await waitFor(() => expect(getSessionMessages).toHaveBeenLastCalledWith('running-chat', undefined))
    expect(await screen.findByText('Transcript for running-chat')).toBeTruthy()

    const header = screen.getByRole('heading', { name: 'Running chat' }).closest('header')
    expect(header).not.toBeNull()
    expect(within(header as HTMLElement).getByText('Running')).toBeTruthy()
  })
})
