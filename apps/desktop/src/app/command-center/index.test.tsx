import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'
import { setSessions } from '@/store/session'

const createWorkPacketFromSession = vi.fn()
const getActionStatus = vi.fn()
const getKanbanBoard = vi.fn()
const getKanbanTask = vi.fn()
const getLogs = vi.fn()
const getStatus = vi.fn()
const getUsageAnalytics = vi.fn()
const restartGateway = vi.fn()
const updateHermes = vi.fn()

vi.mock('@/hermes', () => ({
  createWorkPacketFromSession: (id: string, profile?: null | string, payload?: unknown) =>
    createWorkPacketFromSession(id, profile, payload),
  getActionStatus: (name: string, lines?: number) => getActionStatus(name, lines),
  getKanbanBoard: () => getKanbanBoard(),
  getKanbanTask: (id: string) => getKanbanTask(id),
  getLogs: (params: unknown) => getLogs(params),
  getStatus: () => getStatus(),
  getUsageAnalytics: (days: number) => getUsageAnalytics(days),
  restartGateway: () => restartGateway(),
  updateHermes: () => updateHermes()
}))

function renderCommandCenter(onOpenSession = vi.fn(), initialSection: 'sessions' | 'work-packets' = 'work-packets') {
  return import('./index').then(({ CommandCenterView }) =>
    render(
      <I18nProvider configClient={null}>
        <MemoryRouter initialEntries={['/command-center']}>
          <CommandCenterView
            initialSection={initialSection}
            onClose={vi.fn()}
            onDeleteSession={vi.fn()}
            onOpenSession={onOpenSession}
          />
        </MemoryRouter>
      </I18nProvider>
    )
  )
}

beforeEach(() => {
  setSessions([])
  getActionStatus.mockResolvedValue({ exit_code: 0, lines: [], name: 'noop', pid: 1, running: false })
  getLogs.mockResolvedValue({ lines: [] })
  getStatus.mockResolvedValue({ active_sessions: 0, gateway_running: true, version: 'test' })
  getUsageAnalytics.mockResolvedValue(null)
  restartGateway.mockResolvedValue({ name: 'restart', pid: 1 })
  updateHermes.mockResolvedValue({ name: 'update', pid: 2 })
  createWorkPacketFromSession.mockResolvedValue({ created: true, task: null, work_packets: { count: 1, open_count: 1 } })

  getKanbanBoard.mockResolvedValue({
    assignees: ['Hermes'],
    columns: [
      {
        name: 'ready',
        tasks: [
          {
            assignee: 'Hermes',
            created_at: 1_800_000_000,
            id: 'task-linked',
            latest_summary: 'Short card summary',
            priority: 5,
            session_bridge: {
              last_active: 1_800_000_050,
              profile: 'default',
              session_exists: true,
              session_id: 'session-linked',
              source: 'desktop',
              title: 'Founder OS linked session'
            },
            status: 'ready',
            title: 'Investigate FounderOS bridge'
          }
        ]
      }
    ],
    latest_event_id: 42,
    now: 1_800_000_100
  })

  getKanbanTask.mockResolvedValue({
    task: {
      assignee: 'Hermes',
      created_at: 1_800_000_000,
      id: 'task-linked',
      latest_summary: 'Full safe handoff summary',
      priority: 5,
      session_bridge: {
        last_active: 1_800_000_050,
        profile: 'default',
        session_exists: true,
        session_id: 'session-linked',
        source: 'desktop',
        title: 'Founder OS linked session'
      },
      status: 'ready',
      title: 'Investigate FounderOS bridge'
    }
  })
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

describe('CommandCenterView work packets', () => {
  it('opens a selected work packet detail and navigates to its linked session', async () => {
    const onOpenSession = vi.fn()

    await renderCommandCenter(onOpenSession)

    fireEvent.click(await screen.findByText('Investigate FounderOS bridge'))

    await waitFor(() => expect(getKanbanTask).toHaveBeenCalledWith('task-linked'))
    expect(await screen.findByText('Full safe handoff summary')).toBeTruthy()
    expect(screen.queryByText('Full task instructions for the FounderOS bridge.')).toBeNull()

    fireEvent.click(screen.getByRole('button', { name: 'Open linked session' }))

    expect(onOpenSession).toHaveBeenCalledWith('session-linked', 'default')
  })

  it('opens a session row linked work packet detail without opening the session', async () => {
    const onOpenSession = vi.fn()

    setSessions([
      {
        ended_at: null,
        id: 'session-linked',
        input_tokens: 0,
        is_active: false,
        last_active: 1_800_000_050,
        message_count: 3,
        model: null,
        output_tokens: 0,
        preview: null,
        source: 'desktop',
        started_at: 1_800_000_000,
        title: 'Founder OS linked session',
        tool_call_count: 0,
        work_packets: {
          count: 1,
          latest: {
            assignee: 'Hermes',
            id: 'task-linked',
            priority: 5,
            status: 'ready',
            title: 'Investigate FounderOS bridge'
          },
          open_count: 1
        }
      }
    ])

    await renderCommandCenter(onOpenSession, 'sessions')

    const badge = await screen.findByRole('button', { name: 'Work packet details: Investigate FounderOS bridge' })
    expect(badge.textContent).toContain('1/1')
    fireEvent.click(badge)

    await waitFor(() => expect(getKanbanBoard).toHaveBeenCalled())
    await waitFor(() => expect(getKanbanTask).toHaveBeenCalledWith('task-linked'))
    expect(await screen.findByText('Full safe handoff summary')).toBeTruthy()
    expect(screen.queryByText('Full task instructions for the FounderOS bridge.')).toBeNull()
    expect(onOpenSession).not.toHaveBeenCalled()
  })

  it('creates a work packet from an unlinked session and updates the row summary', async () => {
    const onOpenSession = vi.fn()

    createWorkPacketFromSession.mockResolvedValue({
      created: true,
      task: {
        assignee: 'Hermes',
        created_at: 1_800_000_120,
        id: 'task-created',
        priority: 4,
        session_id: 'session-unlinked',
        status: 'ready',
        title: 'Created packet from session'
      },
      work_packets: {
        count: 1,
        latest: {
          assignee: 'Hermes',
          id: 'task-created',
          priority: 4,
          status: 'ready',
          title: 'Created packet from session'
        },
        open_count: 1
      }
    })
    setSessions([
      {
        ended_at: null,
        id: 'session-unlinked',
        input_tokens: 0,
        is_active: false,
        last_active: 1_800_000_110,
        message_count: 2,
        model: null,
        output_tokens: 0,
        preview: null,
        source: 'desktop',
        started_at: 1_800_000_000,
        title: 'Unlinked FounderOS session',
        tool_call_count: 0
      }
    ])

    await renderCommandCenter(onOpenSession, 'sessions')

    fireEvent.click(await screen.findByRole('button', { name: 'Create work packet' }))

    await waitFor(() => expect(createWorkPacketFromSession).toHaveBeenCalledWith('session-unlinked', undefined, undefined))
    await waitFor(() => expect(getKanbanBoard).toHaveBeenCalled())

    const badge = await screen.findByRole('button', { name: 'Work packet details: Created packet from session' })
    expect(badge.textContent).toContain('1/1')
    expect(onOpenSession).not.toHaveBeenCalled()
  })

  it('opens a linked work packet that is outside the recent list', async () => {
    const onOpenSession = vi.fn()

    const hiddenTask = {
      assignee: 'Hermes',
      created_at: 1_800_000_000,
      id: 'task-outside-recent',
      latest_summary: 'Older card summary',
      priority: 4,
      session_bridge: {
        last_active: 1_800_000_050,
        profile: 'default',
        session_exists: true,
        session_id: 'session-outside-recent',
        source: 'desktop',
        title: 'Outside recent linked session'
      },
      status: 'ready',
      title: 'Backfill old FounderOS packet'
    }

    const newerTasks = Array.from({ length: 6 }, (_, index) => ({
      assignee: 'Hermes',
      created_at: 1_800_000_100 + index,
      id: `task-newer-${index}`,
      latest_summary: `Newer summary ${index}`,
      priority: 3,
      session_bridge: {
        last_active: 1_800_000_200 + index,
        profile: 'default',
        session_exists: true,
        session_id: `session-newer-${index}`,
        source: 'desktop',
        title: `Newer linked session ${index}`
      },
      status: 'ready',
      title: `Newer packet ${index}`
    }))

    getKanbanBoard.mockResolvedValue({
      assignees: ['Hermes'],
      columns: [
        {
          name: 'ready',
          tasks: [...newerTasks, hiddenTask]
        }
      ],
      latest_event_id: 43,
      now: 1_800_000_300
    })
    getKanbanTask.mockResolvedValue({
      task: {
        ...hiddenTask,
        latest_summary: 'Full older safe summary'
      }
    })

    await renderCommandCenter(onOpenSession)

    fireEvent.click(await screen.findByRole('button', { name: 'Work packet details: Backfill old FounderOS packet' }))

    await waitFor(() => expect(getKanbanTask).toHaveBeenCalledWith('task-outside-recent'))
    expect(await screen.findByText('Full older safe summary')).toBeTruthy()
    expect(screen.queryByText('Raw old task body should stay hidden.')).toBeNull()

    fireEvent.click(screen.getByRole('button', { name: 'Open linked session' }))

    expect(onOpenSession).toHaveBeenCalledWith('session-outside-recent', 'default')
  })
})
