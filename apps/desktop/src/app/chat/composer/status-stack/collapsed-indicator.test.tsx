import { cleanup, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest'

import { setComposerTodosVisible } from '@/store/composer-todos-visible'
import { $todosBySession } from '@/store/todos'

import { ComposerStatusStack } from './index'

describe('ComposerStatusStack todo visibility', () => {
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
    setComposerTodosVisible(true)
    $todosBySession.set({})
  })

  it('renders todo items from the todo store by default', () => {
    $todosBySession.set({
      'session-1': [{ content: 'Wire the status stack', id: '1', status: 'in_progress' }]
    })

    render(
      <MemoryRouter>
        <ComposerStatusStack queue={null} sessionId="session-1" />
      </MemoryRouter>
    )

    expect(screen.getByText('Wire the status stack')).toBeTruthy()
    expect(screen.getByText('Tasks 0/1')).toBeTruthy()
    expect(screen.getAllByRole('status').length).toBeGreaterThan(0)
  })

  it('hides todo items when the composer task list toggle is off', () => {
    setComposerTodosVisible(false)
    $todosBySession.set({
      'session-1': [{ content: 'Wire the status stack', id: '1', status: 'in_progress' }]
    })

    render(
      <MemoryRouter>
        <ComposerStatusStack queue={null} sessionId="session-1" />
      </MemoryRouter>
    )

    expect(screen.queryByText('Wire the status stack')).toBeNull()
    expect(screen.queryByText('Tasks 0/1')).toBeNull()
    expect(screen.queryByRole('status')).toBeNull()
  })
})
