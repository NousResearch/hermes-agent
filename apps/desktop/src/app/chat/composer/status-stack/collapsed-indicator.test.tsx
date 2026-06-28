import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest'

import { $todosBySession } from '@/store/todos'

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
    $todosBySession.set({})
  })

  it('shows a running indicator next to the collapsed todo label', () => {
    $todosBySession.set({
      'session-1': [{ content: 'Wire the status stack', id: '1', status: 'in_progress' }]
    })

    render(
      <MemoryRouter>
        <ComposerStatusStack queue={null} sessionId="session-1" />
      </MemoryRouter>
    )

    const button = screen.getByRole('button', { name: /Tasks 0\/1/ })
    fireEvent.click(button)

    const label = screen.getByText('Tasks 0/1')
    const indicator = screen.getByRole('status')

    expect(screen.queryByText('Wire the status stack')).toBeNull()
    expect(button.contains(indicator)).toBe(true)
    expect(label.compareDocumentPosition(indicator) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
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
