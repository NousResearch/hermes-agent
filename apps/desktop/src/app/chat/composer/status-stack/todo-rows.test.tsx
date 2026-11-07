import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, it } from 'vitest'

import type { ComposerStatusItem, StatusItemState } from '@/store/composer-status'

import { TodoRows } from './todo-rows'

type TodoStatus = NonNullable<ComposerStatusItem['todoStatus']>

const row = (id: string, title: string, todoStatus: TodoStatus, depth: number): ComposerStatusItem => {
  const state: StatusItemState = todoStatus === 'in_progress' ? 'running' : 'done'

  return { depth, id, state, title, todoStatus, type: 'todo' }
}

const items = [
  row('phase-a', 'Phase A', 'completed', 0),
  row('a1', 'A one', 'completed', 1),
  row('a2', 'A two', 'cancelled', 1),
  row('phase-b', 'Phase B', 'in_progress', 0),
  row('b1', 'B live', 'in_progress', 1),
  row('b2', 'B later', 'pending', 1),
  row('alone', 'Standalone', 'pending', 0)
]

afterEach(cleanup)

it('folds a settled phase and keeps the phase holding the live item open', () => {
  render(<TodoRows items={items} />)

  expect(screen.getByText('Phase A')).toBeDefined()
  expect(screen.queryByText('A one')).toBeNull()
  expect(screen.queryByText('A two')).toBeNull()
  expect(screen.getByText('B live')).toBeDefined()
  expect(screen.getByText('B later')).toBeDefined()
  expect(screen.getByText('Standalone')).toBeDefined()
})

it('exposes the disclosure state and expands on a row click', () => {
  render(<TodoRows items={items} />)

  const phaseA = screen.getByText('Phase A').closest('[data-slot="status-row"]') as HTMLElement

  expect(phaseA.getAttribute('aria-expanded')).toBe('false')
  fireEvent.click(phaseA)
  expect(phaseA.getAttribute('aria-expanded')).toBe('true')
  expect(screen.getByText('A one')).toBeDefined()
  expect(screen.getByText('A two')).toBeDefined()
})

it('prints the child progress on a parent row', () => {
  render(<TodoRows items={items} />)

  expect(screen.getByText('2/2')).toBeDefined()
  expect(screen.getByText('0/2')).toBeDefined()
})

it('hides a whole subtree, not just one level', () => {
  const nested = [
    row('p', 'Parent', 'completed', 0),
    row('c', 'Child', 'completed', 1),
    row('g', 'Grandchild', 'completed', 2)
  ]

  render(<TodoRows items={nested} />)
  expect(screen.queryByText('Child')).toBeNull()
  expect(screen.queryByText('Grandchild')).toBeNull()

  fireEvent.click(screen.getByText('Parent').closest('[data-slot="status-row"]') as HTMLElement)
  expect(screen.getByText('Child')).toBeDefined()
  // The child is settled too, so it folds its own grandchild.
  expect(screen.queryByText('Grandchild')).toBeNull()
})

it('leaves a leaf without a disclosure', () => {
  render(<TodoRows items={items} />)

  const alone = screen.getByText('Standalone').closest('[data-slot="status-row"]') as HTMLElement

  expect(alone.getAttribute('aria-expanded')).toBeNull()
})
