import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest'

import { createClientSessionState } from '@/lib/chat-runtime'
import { $sessionStates } from '@/store/session-states'
import { $todoContinuationsBySession, $todosBySession } from '@/store/todos'

import { ComposerStatusStack } from './index'

describe('ComposerStatusStack collapsed todo indicator', () => {
  beforeAll(() => {
    vi.stubGlobal(
      'ResizeObserver',
      class {
        disconnect() {}
        observe() {}
      }
    )
  })

  afterEach(() => {
    cleanup()
    $sessionStates.set({})
    $todoContinuationsBySession.set({})
    $todosBySession.set({})
  })

  it('shows a running indicator while the todo group is expanded', () => {
    $sessionStates.set({
      'session-1': { ...createClientSessionState(), busy: true, turnLive: true }
    })
    $todosBySession.set({
      'session-1': [{ content: 'Wire the status stack', id: '1', status: 'in_progress' }]
    })

    render(
      <MemoryRouter>
        <ComposerStatusStack queue={null} sessionId="session-1" />
      </MemoryRouter>
    )

    expect(screen.getByText('Wire the status stack')).toBeTruthy()
    expect(screen.getAllByRole('status').length).toBeGreaterThan(0)
    expect(screen.getByText('Running — Tasks 0/1')).toBeTruthy()
  })

  it('shows a running indicator next to the collapsed todo label', () => {
    $sessionStates.set({
      'session-1': { ...createClientSessionState(), busy: true, turnLive: true }
    })
    $todosBySession.set({
      'session-1': [{ content: 'Wire the status stack', id: '1', status: 'in_progress' }]
    })

    render(
      <MemoryRouter>
        <ComposerStatusStack queue={null} sessionId="session-1" />
      </MemoryRouter>
    )

    const button = screen.getByRole('button', { name: /Running — Tasks 0\/1/ })
    fireEvent.click(button)

    const label = screen.getByText('Running — Tasks 0/1')
    const indicator = screen.getByRole('status')

    expect(screen.queryByText('Wire the status stack')).toBeNull()
    expect(button.contains(indicator)).toBe(true)
    expect(label.compareDocumentPosition(indicator) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
  })

  it('shows authoritative continuation and pause labels without a liveness spinner', () => {
    $todosBySession.set({
      'session-1': [
        { content: 'Finished task', id: '1', status: 'completed' },
        { content: 'Remaining task', id: '2', status: 'in_progress' }
      ]
    })
    $todoContinuationsBySession.set({ 'session-1': { revision: 7, state: 'active' } })

    const view = render(
      <MemoryRouter>
        <ComposerStatusStack queue={null} sessionId="session-1" />
      </MemoryRouter>
    )

    expect(screen.getByText('Goal active — Tasks 1/2')).toBeTruthy()
    expect(screen.queryByRole('status')).toBeNull()

    act(() => {
      $todoContinuationsBySession.set({
        'session-1': { revision: 8, state: 'paused', stopReason: 'Turn budget exhausted' }
      })
    })

    expect(screen.getByText('Goal paused — Tasks 1/2')).toBeTruthy()
    expect(screen.getByText('Turn budget exhausted')).toBeTruthy()
    expect(screen.queryByRole('status')).toBeNull()
    view.unmount()
  })

  it('does not show stale unfinished rows after a completed turn with no authoritative goal', () => {
    $todosBySession.set({
      'session-1': [{ content: 'Stale active task', id: '1', status: 'in_progress' }]
    })

    const view = render(
      <MemoryRouter>
        <ComposerStatusStack queue={null} sessionId="session-1" />
      </MemoryRouter>
    )

    expect(view.container.firstChild).toBeNull()
  })

  it('does not show a collapsed todo indicator when no todo is running', () => {
    $todosBySession.set({
      'session-1': [{ content: 'Wire the status stack', id: '1', status: 'completed' }]
    })

    render(
      <MemoryRouter>
        <ComposerStatusStack queue={null} sessionId="session-1" />
      </MemoryRouter>
    )

    fireEvent.click(screen.getByRole('button', { name: /Tasks 1\/1/ }))

    expect(screen.queryByText('Wire the status stack')).toBeNull()
    expect(screen.queryByRole('status')).toBeNull()
  })
})
