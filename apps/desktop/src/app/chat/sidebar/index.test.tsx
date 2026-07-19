import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { SidebarProvider } from '@/components/ui/sidebar'
import type { SessionInfo } from '@/hermes'
import { I18nProvider } from '@/i18n'
import { $sidebarProjectsOpen, $sidebarRecentsOpen } from '@/store/layout'
import { $profiles } from '@/store/profile'
import { $projects, $projectTree } from '@/store/projects'
import { $gatewayState, $messagingSessions, $sessions, $sessionsLoading, $sessionsTotal } from '@/store/session'

import { ChatSidebar } from './index'

const session: SessionInfo = {
  ended_at: null,
  id: 'session-1',
  input_tokens: 0,
  is_active: false,
  last_active: 1,
  message_count: 1,
  model: null,
  output_tokens: 0,
  preview: 'A desktop session',
  source: 'desktop',
  started_at: 1,
  title: 'A desktop session',
  tool_call_count: 0
}

function renderSidebar() {
  return render(
    <MemoryRouter>
      <I18nProvider configClient={null}>
        <SidebarProvider>
          <ChatSidebar
            currentView="chat"
            onArchiveSession={vi.fn()}
            onBranchSession={vi.fn()}
            onDeleteSession={vi.fn()}
            onLoadMoreSessions={vi.fn()}
            onManageCronJob={vi.fn()}
            onNavigate={vi.fn()}
            onNewSessionInWorkspace={vi.fn()}
            onResumeSession={vi.fn()}
            onTriggerCronJob={vi.fn()}
          />
        </SidebarProvider>
      </I18nProvider>
    </MemoryRouter>
  )
}

describe('ChatSidebar primary sections', () => {
  beforeEach(() => {
    $gatewayState.set('idle')
    $profiles.set([])
    $projects.set([])
    $projectTree.set([])
    $sessions.set([session])
    $messagingSessions.set([])
    $sessionsLoading.set(false)
    $sessionsTotal.set(1)
    $sidebarProjectsOpen.set(false)
    $sidebarRecentsOpen.set(true)
  })

  afterEach(() => {
    cleanup()
    $sessions.set([])
    $messagingSessions.set([])
  })

  it('keeps Projects and Sessions as independent sections when Projects is expanded', () => {
    renderSidebar()

    expect(screen.getByRole('button', { name: 'Projects' })).toBeTruthy()
    expect(screen.getByRole('button', { name: /^Sessions/ })).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: 'Projects' }))

    expect(screen.getByText('No projects yet — create one with +')).toBeTruthy()
    expect(screen.getByRole('button', { name: /^Sessions/ })).toBeTruthy()
    expect(screen.getByRole('button', { name: 'A desktop session' })).toBeTruthy()
  })

  it('keeps the primary sections visible when only messaging sessions exist', () => {
    $sessions.set([])
    $sessionsTotal.set(0)
    $messagingSessions.set([{ ...session, id: 'telegram-1', source: 'telegram' }])

    renderSidebar()

    expect(screen.getByRole('button', { name: 'Projects' })).toBeTruthy()
    expect(screen.getByRole('button', { name: /^Sessions/ })).toBeTruthy()
    expect(screen.getByRole('button', { name: /^Telegram/ })).toBeTruthy()
  })
})
