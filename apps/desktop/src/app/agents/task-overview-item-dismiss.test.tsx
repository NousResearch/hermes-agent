import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { $sessions } from '@/store/session'
import { $sessionStates } from '@/store/session-states'
import { resetSessionTodoOverview } from '@/store/session-todos-overview'
import { $todosBySession, setSessionTodos } from '@/store/todos'

import { AgentsPanelContent } from './index'

/** Regression: dismissing ONE todo item from a row's checklist must remove
 *  only that item (not the whole row) and repaint live, exactly like row
 *  dismissal already did. */
describe('TaskOverviewItemRow dismiss', () => {
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

  it('removes only the dismissed item, keeps the row and its other items', () => {
    act(() => {
      setSessionTodos('sess-1', [
        { content: '첫번째 항목', id: '1', status: 'completed' },
        { content: '두번째 항목', id: '2', status: 'completed' }
      ])
    })

    render(<AgentsPanelContent />)

    // Completed items linger behind "Show N more finished" — expand it first.
    act(() => {
      fireEvent.click(screen.getByText(/Show \d+ more finished/))
    })

    expect(screen.queryByText('첫번째 항목')).toBeTruthy()
    expect(screen.queryByText('두번째 항목')).toBeTruthy()

    const dismissButtons = screen.getAllByLabelText('Dismiss from this panel')
    expect(dismissButtons.length).toBe(3) // [0] row-level, [1] first item, [2] second item
    // Dismiss the FIRST item specifically (index 1) — the row-level dismiss
    // (index 0) removes the whole row and must be left alone by this test.
    act(() => {
      fireEvent.click(dismissButtons[1]!)
    })

    expect(screen.queryByText('첫번째 항목')).toBeNull()
    expect(screen.queryByText('두번째 항목')).toBeTruthy()
    expect(screen.queryByText('Test Session')).toBeTruthy()
  })
})
