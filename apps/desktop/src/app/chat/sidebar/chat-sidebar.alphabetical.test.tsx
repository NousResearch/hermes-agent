// @vitest-environment jsdom
import { cleanup, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, describe, expect, it } from 'vitest'

import { SidebarProvider } from '@/components/ui/sidebar'
import { $pinnedSessionIds, setSidebarAgentsGrouped } from '@/store/layout'
import { $projectScope, ALL_PROJECTS } from '@/store/project-scope'
import { $projectTree } from '@/store/projects'
import { $sessions, $sessionsLoading } from '@/store/session'
import { setSidebarAlphabetical } from '@/store/sidebar-alphabetical'
import { makeSessionInfo } from '@/test/session-info'

import { ChatSidebar } from './index'

const noop = () => {}

const noopAsync = async () => {}

const renderSidebar = () =>
  render(
    <MemoryRouter initialEntries={['/']}>
      <SidebarProvider>
        <ChatSidebar
          currentView="chat"
          onArchiveSession={noop}
          onBranchSession={noop}
          onDeleteSession={noop}
          onLoadMoreSessions={noop}
          onManageCronJob={noop}
          onNavigate={noop}
          onNewSessionInWorkspace={noop}
          onNewSessionSplit={noop}
          onResumeSession={noop}
          onRetrySessions={noopAsync}
          onTriggerCronJob={noopAsync}
        />
      </SidebarProvider>
    </MemoryRouter>
  )

// DOM order of the given visible texts, so a test asserts what the user sees.
const domOrder = (labels: string[]) =>
  labels
    .map(label => ({ label, node: screen.getByText(label) }))
    .sort((a, b) => (a.node.compareDocumentPosition(b.node) & Node.DOCUMENT_POSITION_FOLLOWING ? -1 : 1))
    .map(entry => entry.label)

afterEach(() => {
  cleanup()
  setSidebarAlphabetical(false)
  setSidebarAgentsGrouped(false)
  $projectScope.set(ALL_PROJECTS)
  $projectTree.set([])
  $pinnedSessionIds.set([])
  $sessions.set([])
  $sessionsLoading.set(true)
})

describe('ChatSidebar "Sort sidebar alphabetically"', () => {
  const project = (id: string, label: string, lastActive: number) => ({
    id: `/repos/${id}`,
    label,
    lastActive,
    path: `/repos/${id}`,
    repos: [],
    sessionCount: 1
  })

  describe('Pinned section', () => {
    const seed = () => {
      $sessions.set([
        makeSessionInfo({ id: 'p-zulu', last_active: 3, started_at: 1, title: 'Zulu notes' }),
        makeSessionInfo({ id: 'p-alpha', last_active: 2, started_at: 1, title: 'Alpha notes' }),
        makeSessionInfo({ id: 'p-mango', last_active: 1, started_at: 1, title: 'Mango notes' })
      ])
      $pinnedSessionIds.set(['p-zulu', 'p-alpha', 'p-mango'])
    }

    it('keeps the stored pin order by default', () => {
      seed()
      renderSidebar()

      expect(domOrder(['Zulu notes', 'Alpha notes', 'Mango notes'])).toEqual(['Zulu notes', 'Alpha notes', 'Mango notes'])
    })

    it('orders pinned chats A→Z when on', () => {
      seed()
      setSidebarAlphabetical(true)
      renderSidebar()

      expect(domOrder(['Zulu notes', 'Alpha notes', 'Mango notes'])).toEqual(['Alpha notes', 'Mango notes', 'Zulu notes'])
    })
  })

  describe('Projects overview', () => {
    const seed = () => {
      setSidebarAgentsGrouped(true)
      $sessionsLoading.set(false)
      $projectTree.set([project('zulu', 'zulu-app', 30), project('alpha', 'alpha-app', 10), project('mango', 'mango-app', 20)])
    }

    it('orders by recent activity by default', () => {
      seed()
      renderSidebar()

      expect(domOrder(['zulu-app', 'alpha-app', 'mango-app'])).toEqual(['zulu-app', 'mango-app', 'alpha-app'])
    })

    it('orders projects A→Z when on', () => {
      seed()
      setSidebarAlphabetical(true)
      renderSidebar()

      expect(domOrder(['zulu-app', 'alpha-app', 'mango-app'])).toEqual(['alpha-app', 'mango-app', 'zulu-app'])
    })
  })
})
