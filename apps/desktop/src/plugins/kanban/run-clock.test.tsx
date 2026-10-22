import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// Test harness supplies the host's locale registration, as plugin loading does.
// eslint-disable-next-line no-restricted-imports
import { registerPluginLocales } from '@/i18n/plugin-i18n'

import { en, KANBAN_LOCALES } from './i18n'
import type { KanbanTask } from './types'
import { RunClock } from './ui'

let disposeLocales: () => void

beforeEach(() => {
  vi.useFakeTimers()
  vi.setSystemTime(new Date('2026-09-16T18:00:00Z'))
  disposeLocales = registerPluginLocales('kanban', KANBAN_LOCALES)
})

afterEach(() => {
  cleanup()
  disposeLocales()
  vi.useRealTimers()
})

// #99819: a retried task's first start is hours old while the fresh run is
// minutes old — the clock must tick from the current run, not task.started_at.
describe('RunClock', () => {
  it('ticks from the current run start instead of the task first start', () => {
    const now = Math.floor(Date.now() / 1000)

    const task: KanbanTask = {
      id: 't1',
      title: 'retried',
      status: 'running',
      started_at: now - 7200, // first start: 2h ago
      current_run_started_at: now - 90 // retry run: 90s ago
    }

    render(<RunClock task={task} />)

    expect(screen.getByText(`${en.working} · 1m`)).toBeTruthy()
  })

  it('falls back to started_at when the backend predates the field', () => {
    const now = Math.floor(Date.now() / 1000)

    const task: KanbanTask = {
      id: 't1',
      title: 'legacy',
      status: 'running',
      started_at: now - 7200
    }

    render(<RunClock task={task} />)

    expect(screen.getByText(`${en.working} · 2h`)).toBeTruthy()
  })

  it('renders nothing when neither timestamp is present', () => {
    const { container } = render(<RunClock task={{ id: 't1', title: 'bare', status: 'running' }} />)

    expect(container.textContent).toBe('')
  })
})
