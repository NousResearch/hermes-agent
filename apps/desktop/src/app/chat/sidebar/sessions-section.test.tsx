import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import type { SessionInfo } from '@/hermes'
import { sessionIdentityKey } from '@/lib/session-identity'

import { SidebarSessionsSection } from './sessions-section'

const crossProfileSession: SessionInfo = {
  ended_at: null,
  id: 'telegram-session',
  input_tokens: 0,
  is_active: false,
  last_active: 1,
  message_count: 2,
  model: null,
  output_tokens: 0,
  preview: 'remote thread',
  profile: 'ubuntu-server',
  source: 'telegram',
  started_at: 1,
  title: 'Cross Profile Chat',
  tool_call_count: 0
}

describe('SidebarSessionsSection profile routing', () => {
  it('passes the session owner when a row is resumed', () => {
    const onResumeSession = vi.fn()

    render(
      <SidebarSessionsSection
        activeSessionId={null}
        emptyState={null}
        label="Sessions"
        onArchiveSession={vi.fn()}
        onDeleteSession={vi.fn()}
        onResumeSession={onResumeSession}
        onToggle={vi.fn()}
        onTogglePin={vi.fn()}
        open
        pinned={false}
        sessions={[crossProfileSession]}
        workingSessionIdSet={new Set()}
      />
    )

    fireEvent.click(screen.getByText('Cross Profile Chat'))

    expect(onResumeSession).toHaveBeenCalledWith('telegram-session', 'ubuntu-server')
  })

  it('isolates selected and working state for colliding stored ids', () => {
    const alpha = { ...crossProfileSession, id: 'shared', profile: 'alpha', title: 'Alpha Chat' }
    const beta = { ...crossProfileSession, id: 'shared', profile: 'beta', title: 'Beta Chat' }

    render(
      <SidebarSessionsSection
        activeSessionId={sessionIdentityKey('shared', 'beta')}
        emptyState={null}
        label="Sessions"
        onArchiveSession={vi.fn()}
        onDeleteSession={vi.fn()}
        onResumeSession={vi.fn()}
        onToggle={vi.fn()}
        onTogglePin={vi.fn()}
        open
        pinned={false}
        sessions={[alpha, beta]}
        workingSessionIdSet={new Set([sessionIdentityKey('shared', 'alpha')])}
      />
    )

    expect(screen.getByText('Alpha Chat').closest('[data-working]')?.getAttribute('data-working')).toBe('true')
    expect(screen.getByText('Alpha Chat').closest('[aria-current]')).toBeNull()
    expect(screen.getByText('Beta Chat').closest('[data-working]')).toBeNull()
    expect(screen.getByText('Beta Chat').closest('[aria-current]')?.getAttribute('aria-current')).toBe('page')
  })
})
