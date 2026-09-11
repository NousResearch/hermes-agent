import { act, cleanup, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { $sessions } from '@/store/session'
import { $sessionStates } from '@/store/session-states'
import { resetSessionTodoOverview } from '@/store/session-todos-overview'
import { $todosBySession, setSessionTodos } from '@/store/todos'

import { AgentsPanelContent } from './index'

/**
 * Reproduction + regression guard: does the Task Overview section re-render
 * LIVE while it is already mounted (panel open), or only on remount (panel
 * closed then reopened)? User report: "closing and reopening the sidebar
 * makes it appear" — i.e. data was captured (module-level subscribe worked)
 * but an already-mounted panel never repainted on a NEW todo.updated
 * snapshot. Root cause: the panel subscribed to an unrelated "tick" counter
 * atom and re-derived the real data through a plain function call on every
 * render — an indirection where the counter reliably changed (confirmed by
 * an independent `.listen()` firing) but React never repainted with the
 * fresh payload. Fixed by putting the actual rows array in the atom itself.
 */
describe('TaskOverviewSection live update while already mounted', () => {
  beforeEach(() => {
    resetSessionTodoOverview()
    $todosBySession.set({})
    $sessionStates.set({})
    $sessions.set([{ id: 'sess-1', preview: '', title: 'Test Session' } as never])
  })

  afterEach(() => {
    cleanup()
    resetSessionTodoOverview()
    $todosBySession.set({})
  })

  it('shows a NEW todo list without unmount/remount when the panel is already open', () => {
    // Panel opens with NOTHING yet — mirrors "no todo has ever fired".
    render(<AgentsPanelContent />)

    expect(screen.queryByText('Test Session')).toBeNull()

    // A todo_list tool call happens WHILE the panel stays mounted (no re-render
    // of AgentsPanelContent itself, no key change) — same as the live desktop
    // app receiving a todo.updated event over the gateway while Agents is open.
    act(() => {
      setSessionTodos('sess-1', [{ content: '첫번째 항목', id: '1', status: 'pending' }])
    })

    // If this fails, the row never appears without a close/reopen (remount) —
    // confirming the live-update bug the user reported.
    expect(screen.queryByText('Test Session')).toBeTruthy()
    expect(screen.queryByText('첫번째 항목')).toBeTruthy()
  })
})
