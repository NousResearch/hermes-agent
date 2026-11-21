import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import type { TodoItem } from '@/lib/todos'
import { setBusy } from '@/store/session'
import {
  $retainedTodosExpanded,
  clearAllSessionTodos,
  restoreSessionTodosFromSnapshot,
  setSessionTodos
} from '@/store/todos'

import { useSessionStatusPresence } from '../hooks/use-status-presence'

import { ComposerStatusStack } from './index'

// A settled checklist: 15 completed and 6 cancelled (superseded) items.
const settled: TodoItem[] = Array.from({ length: 21 }, (_, i) => ({
  content: `Synthetic step ${i + 1}`,
  id: `step-${i + 1}`,
  status: i < 15 ? 'completed' : 'cancelled'
}))

const snapshot = { revision: 26, todos: settled }

function Presence({ sessionId }: { sessionId: string }) {
  return <output data-testid="presence">{String(useSessionStatusPresence(sessionId))}</output>
}

const stack = (sessionId: string) => (
  <MemoryRouter>
    <ComposerStatusStack queue={null} sessionId={sessionId} />
    <Presence sessionId={sessionId} />
  </MemoryRouter>
)

const history = (done = 15, total = 21) => screen.queryByRole('button', { name: `Previous tasks ${done}/${total}` })

beforeEach(() => vi.useFakeTimers())

afterEach(() => {
  cleanup()
  setBusy(false)
  clearAllSessionTodos()
  $retainedTodosExpanded.set({})
  vi.useRealTimers()
})

// What the gateway and hydration paths do to an idle, settled conversation.
const transitions: [string, () => void][] = [
  ['a prompt starts the turn', () => setBusy(true)],
  ['the turn re-reads its settled list (equal revision)', () => restoreSessionTodosFromSnapshot('a', snapshot, true)],
  ['the turn completes', () => setBusy(false)],
  [
    'activation or post-turn hydration replays the snapshot',
    () => restoreSessionTodosFromSnapshot('a', snapshot, false)
  ],
  ['a finished-list linger would elapse', () => vi.advanceTimersByTime(5_000)]
]

it.each([
  ['collapsed', 2],
  ['expanded', 1]
])('a settled checklist stays one quiet %s disclosure through ordinary turns', (choice, clicks) => {
  act(() => restoreSessionTodosFromSnapshot('a', snapshot, false))
  render(stack('a'))

  const header = history()
  expect(header).not.toBeNull()

  for (let i = 0; i < clicks; i += 1) {
    fireEvent.click(header!)
  }

  const expanded = String(choice === 'expanded')

  for (const [step, apply] of transitions) {
    act(apply)

    // Same node: no unmount/remount; same aria state: the user's choice holds.
    expect(history(), step).toBe(header)
    expect(header!.getAttribute('aria-expanded'), step).toBe(expanded)
    expect(screen.queryByRole('button', { name: /^Tasks / }), step).toBeNull()
    expect(screen.queryAllByText(/^Synthetic step /).length, step).toBe(choice === 'expanded' ? 21 : 0)
    expect(screen.getByTestId('presence').textContent, step).toBe('true')
  }
})

it('keeps disclosure choices per session and still hands a new plan to the live feed', () => {
  act(() => {
    restoreSessionTodosFromSnapshot('a', snapshot, false)
    restoreSessionTodosFromSnapshot('b', { revision: 4, todos: settled.slice(0, 2) }, false)
  })
  const view = render(stack('a'))

  fireEvent.click(history()!)
  view.rerender(stack('b'))
  expect(history(2, 2)?.getAttribute('aria-expanded')).toBe('false')
  view.rerender(stack('a'))
  expect(history()?.getAttribute('aria-expanded')).toBe('true')

  // A real new plan is live progress: it replaces the history row while it runs.
  act(() => {
    setBusy(true)
    setSessionTodos('a', [{ content: 'Synthetic new step', id: 'new', status: 'in_progress' }], 27)
  })
  expect(screen.getByRole('button', { name: /^Tasks 0\/1/ }).getAttribute('aria-expanded')).toBe('true')
  expect(history()).toBeNull()
})
