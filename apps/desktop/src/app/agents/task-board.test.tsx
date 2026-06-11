import { cleanup, render, screen } from '@testing-library/react'
import type { ReactNode } from 'react'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { I18nProvider } from '@/i18n'
import { $tasksBySession, TASK_LINGER_MS, upsertAgentTask } from '@/store/tasks'

import { TaskBoard } from './task-board'

function wrapper({ children }: { children: ReactNode }) {
  return <I18nProvider initialLocale="en">{children}</I18nProvider>
}

const nowSeconds = () => Date.now() / 1000

describe('TaskBoard', () => {
  beforeEach(() => $tasksBySession.set({}))

  afterEach(() => {
    cleanup()
    $tasksBySession.set({})
  })

  it('renders nothing while the registry is quiet', () => {
    const { container } = render(<TaskBoard sessionId="s1" />, { wrapper })

    expect(container.querySelector('[data-slot="task-board"]')).toBeNull()
  })

  it('shows one row per active task with intent badge, label, and tool meta', () => {
    upsertAgentTask('s1', {
      goal: 'Find the release notes',
      intent: 'web_research',
      label: 'Release notes hunt',
      last_tool: 'fetch_url',
      session_id: 's1',
      started_at: nowSeconds() - 10,
      status: 'running',
      task_id: 'sa-0-abcd1234',
      tool_count: 3
    })

    render(<TaskBoard sessionId="s1" />, { wrapper })

    expect(screen.getByText('Tasks')).toBeTruthy()
    expect(screen.getByText('web_research')).toBeTruthy()
    expect(screen.getByText('Release notes hunt')).toBeTruthy()
    expect(screen.getByText(/3 tools · fetch_url/)).toBeTruthy()
  })

  it('only shows tasks for the active session', () => {
    upsertAgentTask('s1', {
      goal: 'Visible work',
      session_id: 's1',
      started_at: nowSeconds() - 10,
      status: 'running',
      task_id: 'visible-task'
    })
    upsertAgentTask('s2', {
      goal: 'Other session work',
      session_id: 's2',
      started_at: nowSeconds() - 5,
      status: 'running',
      task_id: 'hidden-task'
    })

    render(<TaskBoard sessionId="s1" />, { wrapper })

    expect(screen.getByText('Visible work')).toBeTruthy()
    expect(screen.queryByText('Other session work')).toBeNull()
  })

  it('drops terminal rows once they age past the linger window', () => {
    upsertAgentTask('s1', {
      finished_at: nowSeconds() - (TASK_LINGER_MS / 1000 + 5),
      goal: 'Old work',
      session_id: 's1',
      started_at: nowSeconds() - 120,
      status: 'succeeded',
      task_id: 'sa-0-old'
    })
    upsertAgentTask('s1', {
      finished_at: nowSeconds() - 1,
      goal: 'Fresh work',
      session_id: 's1',
      started_at: nowSeconds() - 30,
      status: 'failed',
      task_id: 'sa-1-fresh'
    })

    render(<TaskBoard sessionId="s1" />, { wrapper })

    expect(screen.queryByText('Old work')).toBeNull()
    expect(screen.getByText('Fresh work')).toBeTruthy()
    expect(screen.getByText(/failed/)).toBeTruthy()
  })
})
