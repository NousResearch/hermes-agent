import { cleanup, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $attentionSessionIds, $sessions, $unreadFinishedSessionIds, $workingSessionIds } from '@/store/session'
import { $switcherIndex, $switcherOpen, $switcherSessions } from '@/store/session-switcher'
import { $subagentsBySession, upsertSubagent } from '@/store/subagents'
import type { SessionInfo } from '@/types/hermes'

import { SessionSwitcher } from './session-switcher'

const session = (id: string, title: string): SessionInfo => ({ id, title }) as SessionInfo

describe('SessionSwitcher background review activity', () => {
  beforeEach(() => {
    Element.prototype.scrollIntoView = vi.fn()
    $attentionSessionIds.set([])
    $sessions.set([])
    $unreadFinishedSessionIds.set([])
    $workingSessionIds.set([])
    $subagentsBySession.set({})
    $switcherSessions.set([session('parent', 'Parent session')])
    $switcherIndex.set(0)
    $switcherOpen.set(true)
  })

  afterEach(() => {
    cleanup()
    $switcherOpen.set(false)
    $switcherSessions.set([])
    $unreadFinishedSessionIds.set([])
    $subagentsBySession.set({})
    vi.restoreAllMocks()
  })

  it('keeps the parent row working while its independent reviewer is running', () => {
    upsertSubagent(
      'runtime-parent',
      { child_session_id: 'review-child', status: 'running', subagent_id: 'review' },
      true,
      'subagent.start',
      'parent'
    )

    render(
      <MemoryRouter>
        <SessionSwitcher />
      </MemoryRouter>
    )

    const row = screen.getByText('Parent session').closest('[data-working]')
    const dot = row?.querySelector('[aria-hidden="true"]')

    expect(row?.getAttribute('data-working')).toBe('true')
    expect(screen.getByText('Session running').classList.contains('sr-only')).toBe(true)
    expect(dot?.classList.contains('rounded-full')).toBe(true)
    expect(dot?.classList.contains('border')).toBe(true)
  })

  it('announces needs-input priority and keeps a non-color diamond signal', () => {
    $attentionSessionIds.set(['parent'])
    $unreadFinishedSessionIds.set(['parent'])

    render(
      <MemoryRouter>
        <SessionSwitcher />
      </MemoryRouter>
    )

    const row = screen.getByText('Parent session').closest('.row-hover')
    const dot = row?.querySelector('[aria-hidden="true"]')

    expect(screen.getByText('Needs input').classList.contains('sr-only')).toBe(true)
    expect(dot?.classList.contains('rotate-45')).toBe(true)
    expect(screen.queryByText('New result, not viewed')).toBeNull()
  })
})
