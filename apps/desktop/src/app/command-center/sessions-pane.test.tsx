import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { SessionInfo } from '@/hermes'

import { CommandCenterView } from '.'

const listAllProfileSessions = vi.hoisted(() => vi.fn())

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  listAllProfileSessions: (...args: unknown[]) => listAllProfileSessions(...args)
}))

const worker = (id: string, profile: string, taskId: string, active: boolean): SessionInfo =>
  ({
    ended_at: active ? null : 120,
    id,
    input_tokens: 0,
    is_active: active,
    kanban_board: 'xscale-control-room',
    kanban_run_status: active ? 'running' : 'done',
    kanban_task_id: taskId,
    kanban_task_status: active ? 'running' : 'done',
    kanban_task_title: active ? 'Client Ops delivery' : 'Agent Ops delivery',
    last_active: active ? 200 : 150,
    message_count: 4,
    model: 'gpt-5.6',
    output_tokens: 0,
    preview: `work kanban task ${taskId}`,
    profile,
    source: 'kanban',
    started_at: 100,
    title: `Work kanban task ${taskId}`,
    tool_call_count: 0
  }) as SessionInfo

describe('Command Center cross-profile worker sessions', () => {
  beforeEach(() => {
    listAllProfileSessions.mockReset()
  })

  it('renders worker task/profile/status, filters profiles, and opens the selected owner row', async () => {
    const client = worker('20260902_164119_254ae5', 'clientops', 't_5faae42d', true)
    const agent = worker('20260902_164323_4b026f', 'agentops', 't_3b5c5c01', false)

    const foreground = {
      ...worker('foreground', 'default', 'ignored', false),
      source: 'desktop',
      title: 'foreground',
      kanban_task_title: undefined
    }

    delete foreground.kanban_task_id
    listAllProfileSessions.mockResolvedValue({ errors: [], sessions: [client, agent, foreground], total: 3 })
    const onOpenSession = vi.fn()

    render(
      <MemoryRouter>
        <CommandCenterView
          initialSection="sessions"
          onClose={() => {}}
          onDeleteSession={() => Promise.resolve()}
          onOpenSession={onOpenSession}
        />
      </MemoryRouter>
    )

    expect(await screen.findByText('t_5faae42d')).toBeTruthy()
    expect(screen.getAllByText('clientops').length).toBeGreaterThan(0)
    expect(screen.getByText('running')).toBeTruthy()
    expect(screen.getByText('t_3b5c5c01')).toBeTruthy()
    expect(screen.getAllByText('agentops').length).toBeGreaterThan(0)
    expect(screen.getByText('done')).toBeTruthy()
    expect(screen.getByText('foreground')).toBeTruthy()

    fireEvent.change(screen.getByRole('combobox', { name: 'All profiles' }), { target: { value: 'clientops' } })
    expect(screen.queryByText('t_3b5c5c01')).toBeNull()
    expect(screen.getByText('t_5faae42d')).toBeTruthy()

    fireEvent.click(screen.getByText('Client Ops delivery'))
    await waitFor(() => expect(onOpenSession).toHaveBeenCalledWith(client))
  })
})
