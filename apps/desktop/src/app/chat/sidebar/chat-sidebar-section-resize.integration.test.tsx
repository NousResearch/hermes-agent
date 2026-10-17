// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { SidebarProvider } from '@/components/ui/sidebar'
import { $connectionsRegistry } from '@/store/connection-registry-state'
import { $cronJobs } from '@/store/cron'
import { setInterfaceMode } from '@/store/interface-mode'
import {
  $sidebarMessagingOpenIds,
  setSidebarAgentsGrouped,
  setSidebarPinsOpen,
  setSidebarRecentsOpen
} from '@/store/layout'
import { $paneStates, setPaneHeightOverride } from '@/store/panes'
import { $profiles } from '@/store/profile'
import {
  $messagingSessions,
  $messagingTruncated,
  $sessionProfilesTruncated,
  $sessions,
  $sessionsLoading
} from '@/store/session'
import { makeSessionInfo } from '@/test/session-info'
import type { CronJob } from '@/types/hermes'

import { SIDEBAR_BOTTOM_SECTION_ID, SIDEBAR_PINNED_SECTION_ID, SIDEBAR_SESSIONS_SECTION_ID } from './section-resize'
import { BOTTOM_H_VAR, PINNED_H_VAR, SESSIONS_MIN_H_VAR } from './section-sash'

import { ChatSidebar } from './index'

const noop = () => {}

const noopAsync = async () => {}

const PINNED_SEAM = 'Resize Pinned and Sessions'
const BOTTOM_SEAM = 'Resize Sessions'

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

const session = (id: string, extra: Parameters<typeof makeSessionInfo>[0] = {}) =>
  makeSessionInfo({ id, last_active: 10, profile: 'default', started_at: 1, title: id, ...extra })

const telegramThread = (id: string) =>
  makeSessionInfo({ connection_id: 'local', id, last_active: 5, profile: 'default', source: 'telegram', title: id })

const CRON_JOB = { enabled: true, id: 'job-1', name: 'nightly', schedule: '* * * * *', state: 'scheduled' } as CronJob

const seam = (name: string) => screen.queryByRole('separator', { name })
const sectionList = (container: HTMLElement) => container.querySelector<HTMLElement>('[data-sessions-mode]')!

const section = (container: HTMLElement, id: string) =>
  container.querySelector<HTMLElement>(`[data-sidebar-section="${id}"]`)!

// a precedes b in document order
const precedes = (a: Node, b: Node) => Boolean(a.compareDocumentPosition(b) & Node.DOCUMENT_POSITION_FOLLOWING)

beforeEach(() => {
  window.localStorage.clear()
  $paneStates.set({})
  $connectionsRegistry.set({
    version: 2,
    primary: 'local',
    secureTokenStorage: true,
    connections: [{ id: 'local', label: 'This computer', kind: 'local', tokenSet: false, tokenPreview: null }]
  } as NonNullable<typeof $connectionsRegistry.value>)
  $profiles.set([{ name: 'default', is_default: true }] as typeof $profiles.value)
  // One pinned row above Sessions, a messaging platform below it, and a
  // further page of sessions on the backend (so load-more shows).
  $sessions.set([session('pinned-one', { pinned: true }), session('recent-one'), session('recent-two')])
  $sessionProfilesTruncated.set({ default: true })
  $messagingSessions.set([telegramThread('tg-one')])
  $messagingTruncated.set(false)
  $sidebarMessagingOpenIds.set(['telegram'])
  $sessionsLoading.set(false)
})

afterEach(() => {
  cleanup()
  setInterfaceMode('advanced')
  setSidebarAgentsGrouped(false)
  setSidebarPinsOpen(true)
  setSidebarRecentsOpen(true)
  $sessions.set([])
  $sessionProfilesTruncated.set({})
  $messagingSessions.set([])
  $sidebarMessagingOpenIds.set([])
  $cronJobs.set([])
  $paneStates.set({})
  window.localStorage.clear()
})

describe('ChatSidebar section resizing', () => {
  it('puts a seam on each side of Sessions, with load-more under the lower one', () => {
    const { container } = renderSidebar()
    const pinnedSeam = seam(PINNED_SEAM)!
    const bottomSeam = seam(BOTTOM_SEAM)!
    const loadMore = screen.getByRole('button', { name: 'Load more' })

    // Pinned → seam → Sessions → seam → load-more → the messaging/cron block.
    const order = [
      section(container, 'pinned'),
      pinnedSeam,
      section(container, 'sessions'),
      bottomSeam,
      loadMore,
      section(container, 'bottom')
    ]

    order.slice(1).forEach((node, i) => expect(precedes(order[i], node)).toBe(true))
    // Load-more still pages Sessions, but sits outside its scroller so the
    // seam can hug the last row and the button stays reachable unscrolled.
    expect(section(container, 'sessions').contains(loadMore)).toBe(false)
    expect(section(container, 'bottom').textContent).toContain('tg-one')
  })

  it('keeps load-more inside Sessions when there is no block below to seam against', () => {
    $messagingSessions.set([])
    $sidebarMessagingOpenIds.set([])
    const { container } = renderSidebar()

    expect(seam(BOTTOM_SEAM)).toBeNull()
    expect(section(container, 'bottom')).toBeNull()
    expect(section(container, 'sessions').contains(screen.getByRole('button', { name: 'Load more' }))).toBe(true)
  })

  it.each([
    {
      name: 'Sessions collapsed',
      setup: () => setSidebarRecentsOpen(false),
      pinned: false,
      bottom: false
    },
    { name: 'Pinned collapsed', setup: () => setSidebarPinsOpen(false), pinned: false, bottom: true },
    {
      name: 'nothing pinned',
      setup: () => $sessions.set([session('recent-one'), session('recent-two')]),
      pinned: false,
      bottom: true
    },
    {
      // The project tree pages its own lanes and hides messaging/cron.
      name: 'grouped by project',
      setup: () => setSidebarAgentsGrouped(true),
      pinned: true,
      bottom: false
    },
    {
      // Cron is advanced chrome: a cron-only block disappears in Simple mode.
      name: 'Simple mode with cron as the only block below',
      setup: () => {
        $messagingSessions.set([])
        $sidebarMessagingOpenIds.set([])
        $cronJobs.set([CRON_JOB])
        setInterfaceMode('simple')
      },
      pinned: true,
      bottom: false
    },
    {
      name: 'Advanced mode with cron as the only block below',
      setup: () => {
        $messagingSessions.set([])
        $sidebarMessagingOpenIds.set([])
        $cronJobs.set([CRON_JOB])
      },
      pinned: true,
      bottom: true
    }
  ])('offers a seam only where both sides can trade height: $name', ({ setup, pinned, bottom }) => {
    act(setup)
    renderSidebar()

    expect(Boolean(seam(PINNED_SEAM))).toBe(pinned)
    expect(Boolean(seam(BOTTOM_SEAM))).toBe(bottom)
  })

  it('drops both seams while searching', () => {
    renderSidebar()

    fireEvent.change(screen.getByPlaceholderText('Search sessions…'), { target: { value: 'recent' } })

    expect(seam(PINNED_SEAM)).toBeNull()
    expect(seam(BOTTOM_SEAM)).toBeNull()
  })

  it('paints saved heights from the pane store, follows later changes, and clears on reset', () => {
    const saved = {
      [SIDEBAR_PINNED_SECTION_ID]: [PINNED_H_VAR, 64],
      [SIDEBAR_SESSIONS_SECTION_ID]: [SESSIONS_MIN_H_VAR, 300],
      [SIDEBAR_BOTTOM_SECTION_ID]: [BOTTOM_H_VAR, 90]
    } as const

    for (const [id, [, px]] of Object.entries(saved)) {
      setPaneHeightOverride(id, px)
    }

    const { container } = renderSidebar()
    const list = sectionList(container)

    for (const [cssVar, px] of Object.values(saved)) {
      expect(list.style.getPropertyValue(cssVar)).toBe(`${px}px`)
    }

    act(() => setPaneHeightOverride(SIDEBAR_BOTTOM_SECTION_ID, 120))
    expect(list.style.getPropertyValue(BOTTOM_H_VAR)).toBe('120px')

    fireEvent.doubleClick(seam(BOTTOM_SEAM)!)

    for (const [cssVar] of Object.values(saved)) {
      expect(list.style.getPropertyValue(cssVar)).toBe('')
    }
  })
})
