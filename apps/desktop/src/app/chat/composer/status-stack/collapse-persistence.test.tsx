import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, expect, it } from 'vitest'

import { $statusSectionCollapsedByKey } from '@/store/status-section-collapse'
import { $todosBySession, clearSessionTodos, restoreSessionTodosFromSnapshot } from '@/store/todos'

import { ComposerStatusStack } from './index'

const stack = (sessionId: string) => (
  <MemoryRouter>
    <ComposerStatusStack queue={null} sessionId={sessionId} />
  </MemoryRouter>
)

const seedTodos = (...sessionIds: string[]) =>
  $todosBySession.set(
    Object.fromEntries(
      sessionIds.map(sessionId => [sessionId, [{ id: 'todo', content: 'Visible todo', status: 'in_progress' }]])
    )
  )

afterEach(() => {
  cleanup()
  $statusSectionCollapsedByKey.set({})
  $todosBySession.set({})
  clearSessionTodos('owner')
  clearSessionTodos('other')
})

it('keeps a manually collapsed todo group collapsed across a full stack remount', () => {
  seedTodos('owner')
  const first = render(stack('owner'))

  expect(screen.getByText('Visible todo')).toBeTruthy()

  fireEvent.click(screen.getByRole('button', { name: /Tasks 0\/1/ }))
  expect(screen.queryByText('Visible todo')).toBeNull()

  // A store-driven render remounts the whole stack (tool tick): the user's
  // collapse must survive the unmount instead of snapping back to default.
  first.unmount()
  render(stack('owner'))

  expect(screen.queryByText('Visible todo')).toBeNull()
  expect(screen.getByRole('button', { name: /Tasks 0\/1/ }).getAttribute('aria-expanded')).toBe('false')
})

it('scopes the collapse choice to the session that made it', () => {
  seedTodos('owner', 'other')
  const view = render(stack('owner'))

  fireEvent.click(screen.getByRole('button', { name: /Tasks 0\/1/ }))
  expect(screen.queryByText('Visible todo')).toBeNull()

  view.rerender(stack('other'))
  expect(screen.getByText('Visible todo')).toBeTruthy()
  expect(screen.getByRole('button', { name: /Tasks 0\/1/ }).getAttribute('aria-expanded')).toBe('true')
})

it('keeps a manual expansion of the collapsed-by-default retained todo list across a remount', () => {
  restoreSessionTodosFromSnapshot(
    'owner',
    {
      revision: 3,
      todos: [
        { id: 'one', content: 'Already done', status: 'completed' },
        { id: 'two', content: 'Still open', status: 'in_progress' }
      ]
    },
    false
  )
  const first = render(stack('owner'))

  const review = screen.getByRole('button', { name: /Previous tasks 1\/2/ })
  expect(review.getAttribute('aria-expanded')).toBe('false')

  fireEvent.click(review)
  expect(screen.getByText('Still open')).toBeTruthy()

  first.unmount()
  render(stack('owner'))

  expect(screen.getByRole('button', { name: /Previous tasks 1\/2/ }).getAttribute('aria-expanded')).toBe('true')
  expect(screen.getByText('Still open')).toBeTruthy()
})
