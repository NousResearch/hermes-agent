import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $liveSessions, clearLiveSessions, reconcileLiveSessions } from '@/store/live-sessions'
import { $activeGatewayProfile, $showAllProfiles } from '@/store/profile'
import { $selectedStoredSessionId } from '@/store/session'
import { $removedSessionIds } from '@/store/session-removal'

import { SidebarLiveSessionsSection } from './live-sessions-section'

afterEach(cleanup)

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      sidebar: {
        row: {
          untitledPlaceholder: 'Untitled session'
        }
      }
    }
  })
}))

const LIVE_ITEM = {
  id: 'runtime-1',
  last_active: 2_000,
  message_count: 0,
  model: 'zyphra/qwen',
  preview: 'first prompt',
  session_key: 'sess-live-a',
  source: 'cli',
  started_at: 1_000,
  status: 'working',
  title: 'Live chat'
} as const

function seedLive(overrides: Record<string, unknown> = {}) {
  return reconcileLiveSessions({ sessions: [{ ...LIVE_ITEM, ...overrides }] }, {
    connectionId: 'conn-1',
    profileKey: 'default'
  })
}

beforeEach(() => {
  clearLiveSessions()
  $liveSessions.set([])
  $selectedStoredSessionId.set(null)
  $removedSessionIds.set(new Set())
  // The sidebar's profile context (`$profileScope` = ALL_PROFILES when "All
  // profiles" is on, else the active gateway's profile).
  $showAllProfiles.set(false)
  $activeGatewayProfile.set('default')
})

describe('SidebarLiveSessionsSection', () => {
  it('renders nothing while no session is live', () => {
    const { container } = render(<SidebarLiveSessionsSection label="Live now" onResumeSession={vi.fn()} />)

    expect(container.innerHTML).toBe('')
  })

  it('renders one row per live session with its title', () => {
    // One snapshot — the reconciler treats each `session.active_list` answer
    // as the full live set, so a second call would REPLACE the first.
    const rows = reconcileLiveSessions(
      {
        sessions: [
          LIVE_ITEM,
          { ...LIVE_ITEM, id: 'runtime-2', session_key: 'sess-live-b', title: 'Second live chat' }
        ]
      },
      { connectionId: 'conn-1', profileKey: 'default' }
    )

    expect(rows).toHaveLength(2)

    render(<SidebarLiveSessionsSection label="Live now" onResumeSession={vi.fn()} />)

    expect(screen.getByText('Live now')).toBeTruthy()
    expect(screen.getByText('Live chat')).toBeTruthy()
    expect(screen.getByText('Second live chat')).toBeTruthy()
  })

  it('falls back to the untitled label when the live session has no title', () => {
    seedLive({ title: '' })

    render(<SidebarLiveSessionsSection label="Live now" onResumeSession={vi.fn()} />)

    expect(screen.getByText('Untitled session')).toBeTruthy()
  })

  it('shows the preview line when the snapshot carries one', () => {
    seedLive()

    render(<SidebarLiveSessionsSection label="Live now" onResumeSession={vi.fn()} />)

    expect(screen.getByText('first prompt')).toBeTruthy()
  })

  it('opens the session with its stored id and the row itself', () => {
    const rows = seedLive()
    const onResumeSession = vi.fn()

    render(<SidebarLiveSessionsSection label="Live now" onResumeSession={onResumeSession} />)

    fireEvent.click(screen.getByText('Live chat'))

    // The row opens through the SAME action a stored row uses, carrying the
    // full row so the owner route (connection+profile stamped by reconcile)
    // resolves without a list lookup — never a no-op menu pretending to act.
    expect(onResumeSession).toHaveBeenCalledWith('sess-live-a', rows[0])
  })

  it('hides a live session owned by another profile while the sidebar is scoped to one', () => {
    seedLive({ profile: 'work' })

    render(<SidebarLiveSessionsSection label="Live now" onResumeSession={vi.fn()} />)

    expect(screen.queryByText('Live chat')).toBeNull()
  })

  it('shows that session once the scope moves to its profile', () => {
    seedLive({ profile: 'work' })
    act(() => $activeGatewayProfile.set('work'))

    render(<SidebarLiveSessionsSection label="Live now" onResumeSession={vi.fn()} />)

    expect(screen.getByText('Live chat')).toBeTruthy()
  })

  it('shows every profile again in the all-profiles view', () => {
    seedLive({ profile: 'work' })
    act(() => $showAllProfiles.set(true))

    render(<SidebarLiveSessionsSection label="Live now" onResumeSession={vi.fn()} />)

    expect(screen.getByText('Live chat')).toBeTruthy()
  })
})
