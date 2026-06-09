import { cleanup, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { SidebarProvider } from '@/components/ui/sidebar'
import { $pinnedSessionIds, $sidebarSessionOrderIds, $sidebarWorkspaceOrderIds } from '@/store/layout'
import { $activeGatewayProfile, $profileColors, $profileOrder, $profiles, $showAllProfiles } from '@/store/profile'
import {
  $cronSessions,
  $attentionSessionIds,
  $selectedStoredSessionId,
  $sessionProfileTotals,
  $sessions,
  $sessionsLoading,
  $sessionsTotal,
  $workingSessionIds
} from '@/store/session'
import type { SessionInfo } from '@/types/hermes'

import { ChatSidebar } from './index'

vi.mock('@/hermes', () => ({
  getProfiles: vi.fn(async () => ({ profiles: [] })),
  searchSessions: vi.fn(async () => ({ results: [] })),
  setApiRequestProfile: vi.fn()
}))

const session: SessionInfo = {
  archived: false,
  cwd: null,
  ended_at: null,
  id: 'session-1',
  _lineage_root_id: null,
  input_tokens: 0,
  is_active: false,
  last_active: 1,
  message_count: 1,
  model: null,
  output_tokens: 0,
  preview: 'Recent session',
  profile: 'default',
  source: 'desktop',
  started_at: 1,
  title: 'Recent session',
  tool_call_count: 0
}

describe('ChatSidebar', () => {
  afterEach(() => {
    cleanup()
  })

  beforeEach(() => {
    $activeGatewayProfile.set('default')
    $attentionSessionIds.set([])
    $cronSessions.set([])
    $pinnedSessionIds.set([])
    $profileColors.set({})
    $profileOrder.set([])
    $profiles.set([
      {
        has_env: false,
        is_default: true,
        model: null,
        name: 'default',
        path: '/Users/test/.hermes',
        provider: null,
        skill_count: 0
      },
      {
        has_env: false,
        is_default: false,
        model: null,
        name: 'work',
        path: '/Users/test/.hermes/profiles/work',
        provider: null,
        skill_count: 0
      }
    ])
    $selectedStoredSessionId.set(null)
    $sessionProfileTotals.set({ default: 1 })
    $sessions.set([session])
    $sessionsLoading.set(false)
    $sessionsTotal.set(1)
    $showAllProfiles.set(false)
    $sidebarSessionOrderIds.set([])
    $sidebarWorkspaceOrderIds.set([])
    $workingSessionIds.set([])
  })

  it('places the profile rail below artifacts and above session search', () => {
    renderSidebar()

    const artifacts = screen.getByText('Artifacts')
    const profiles = screen.getByLabelText('Profiles')
    const search = screen.getByRole('textbox', { name: 'Search sessions' })

    expect(Boolean(artifacts.compareDocumentPosition(profiles) & Node.DOCUMENT_POSITION_FOLLOWING)).toBe(true)
    expect(Boolean(profiles.compareDocumentPosition(search) & Node.DOCUMENT_POSITION_FOLLOWING)).toBe(true)
  })

  it('shows an icon-only needs-input badge on a session row that is waiting for approval', () => {
    $attentionSessionIds.set(['session-1'])
    $workingSessionIds.set(['session-1'])

    const { container } = renderSidebar()

    expect(container.querySelector('[data-slot="session-needs-input-badge"]')).toBeTruthy()
    expect(screen.getByRole('status', { name: 'Needs your input' })).toBeTruthy()
    expect(screen.queryByText('Needs your input')).toBeNull()
  })
})

function renderSidebar() {
  return render(
    <MemoryRouter>
      <SidebarProvider>
        <ChatSidebar
          currentView="chat"
          onArchiveSession={vi.fn()}
          onDeleteSession={vi.fn()}
          onLoadMoreSessions={vi.fn()}
          onManageCronJob={vi.fn()}
          onNavigate={vi.fn()}
          onNewSessionInWorkspace={vi.fn()}
          onResumeSession={vi.fn()}
          onTriggerCronJob={vi.fn()}
        />
      </SidebarProvider>
    </MemoryRouter>
  )
}
