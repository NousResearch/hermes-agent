import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'
import { $attentionSessionIds } from '@/store/session'
import type { SessionInfo } from '@/types/hermes'

import { SidebarSessionRow } from './session-row'

function makeSession(overrides: Partial<SessionInfo> = {}): SessionInfo {
  return {
    archived: false,
    cwd: '/Users/example/Workspaces/hermes-agent',
    ended_at: null,
    id: '20260612_100000_row01',
    input_tokens: 0,
    is_active: false,
    last_active: 1_000,
    message_count: 2,
    model: 'claude',
    output_tokens: 0,
    preview: 'Fix shared session row labels',
    source: 'webhook',
    started_at: 1_000,
    title: 'Shared channel session',
    tool_call_count: 0,
    ...overrides
  }
}

function renderRow(session: SessionInfo) {
  return render(
    <I18nProvider configClient={null}>
      <SidebarSessionRow
        isPinned={false}
        isSelected={false}
        isWorking={false}
        onArchive={vi.fn()}
        onDelete={vi.fn()}
        onPin={vi.fn()}
        onResume={vi.fn()}
        session={session}
      />
    </I18nProvider>
  )
}

describe('SidebarSessionRow', () => {
  afterEach(() => {
    cleanup()
    $attentionSessionIds.set([])
  })

  it('renders safe shared-channel origin labels below the title', () => {
    renderRow(
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
    )

    expect(screen.getByText('Shared channel session')).toBeTruthy()
    expect(screen.getByText('Build Room · Release coordination')).toBeTruthy()
  })

  it('keeps ordinary sessions to the title-only row', () => {
    renderRow(makeSession({ channel_origin: null }))

    expect(screen.getByText('Shared channel session')).toBeTruthy()
    expect(screen.queryByText(/Build Room/)).toBeNull()
  })
})
