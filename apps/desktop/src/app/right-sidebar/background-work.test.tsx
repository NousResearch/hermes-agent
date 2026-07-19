import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'
import type { RailTask } from '@/store/activity'

import { BackgroundWorkPanel } from './background-work'

const tasks: RailTask[] = [
  {
    id: 'session:running',
    kind: 'session',
    label: 'Ship phone handoff',
    sessionId: 'running',
    status: 'running',
    updatedAt: 2_000
  },
  {
    id: 'action:lint',
    kind: 'action',
    label: 'Lint desktop',
    status: 'error',
    updatedAt: 1_000
  }
]

const renderPanel = (items: RailTask[], onOpenSession = vi.fn()) => {
  render(
    <I18nProvider>
      <BackgroundWorkPanel onOpenSession={onOpenSession} tasks={items} />
    </I18nProvider>
  )

  return onOpenSession
}

describe('BackgroundWorkPanel', () => {
  it('separates running and finished work and opens session tasks', () => {
    const onOpenSession = renderPanel(tasks)

    expect(screen.getByRole('heading', { name: 'Running' })).toBeTruthy()
    expect(screen.getByRole('heading', { name: 'Finished' })).toBeTruthy()
    expect(screen.getByText('Ship phone handoff')).toBeTruthy()
    expect(screen.getByText('Lint desktop')).toBeTruthy()
    expect(screen.getByText('Failed')).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: 'Open Ship phone handoff' }))
    expect(onOpenSession).toHaveBeenCalledWith('running')
  })

  it('keeps the rail present when empty and supports keyboard-friendly collapse', () => {
    renderPanel([])

    expect(screen.getByText('No background work')).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: 'Collapse background work' }))

    expect(screen.queryByText('No background work')).toBeNull()
    expect(screen.getByRole('button', { name: 'Expand background work' }).getAttribute('aria-expanded')).toBe('false')
  })
})
