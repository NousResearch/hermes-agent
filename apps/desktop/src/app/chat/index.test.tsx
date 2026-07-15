import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'
import { $pinnedSessionIds } from '@/store/layout'
import { $sessions } from '@/store/session'
import type { SessionInfo } from '@/types/hermes'

import { ChatHeader } from './index'

function makeSession(overrides: Partial<SessionInfo> = {}): SessionInfo {
  return {
    archived: false,
    cwd: '/Users/example/Workspaces/hermes-agent',
    ended_at: null,
    id: 'shared-session',
    input_tokens: 0,
    is_active: false,
    last_active: 1_000,
    message_count: 2,
    model: 'claude',
    output_tokens: 0,
    preview: 'Shared-channel chat',
    source: 'webhook',
    started_at: 1_000,
    title: 'Release fix',
    tool_call_count: 0,
    ...overrides
  }
}

function renderHeader(selectedSessionId = 'shared-session') {
  return render(
    <I18nProvider configClient={null}>
      <ChatHeader
        activeSessionId={null}
        isRoutedSessionView={false}
        onDeleteSelectedSession={vi.fn()}
        onToggleSelectedPin={vi.fn()}
        selectedSessionId={selectedSessionId}
      />
    </I18nProvider>
  )
}

describe('ChatHeader', () => {
  afterEach(() => {
    cleanup()
    $sessions.set([])
    $pinnedSessionIds.set([])
  })

  it('shows safe shared-channel origin context for the open chat', () => {
    $sessions.set([
      makeSession({
        channel_origin: {
          chat_name: 'Build Room',
          chat_topic: 'Release coordination',
          chat_type: 'channel',
          display_name: 'Build Room',
          has_thread: true,
          platform: 'webhook'
        }
      })
    ])

    renderHeader()

    expect(screen.getByText('Release fix')).toBeTruthy()
    expect(screen.getByText('Build Room · Release coordination')).toBeTruthy()
  })

  it('keeps ordinary open chats to the title-only header', () => {
    $sessions.set([makeSession({ channel_origin: null })])

    renderHeader()

    expect(screen.getByText('Release fix')).toBeTruthy()
    expect(screen.queryByText(/Build Room/)).toBeNull()
  })
})
