import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $sessions } from '@/store/session'
import { $sessionStates } from '@/store/session-states'
import { resetSessionTodoOverview } from '@/store/session-todos-overview'
import { $todosBySession, setSessionTodos } from '@/store/todos'

import { AgentsPanelContent } from './index'

vi.mock('@/store/gateway', () => ({ $gateway: { get: () => null } }))

/** Regression: clicking the X on a still-active (pending/in_progress) todo item
 *  must attempt a real cancel (not a silent local hide) — and since the mocked
 *  gateway is disconnected, the optimistic flip must roll back so the item stays
 *  visible with its original status rather than vanishing. */
describe('TaskOverviewItemRow cancel (active item)', () => {
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

  it('shows a Cancel label (not Dismiss) for a pending item, and keeps the row on a failed cancel', async () => {
    act(() => {
      setSessionTodos('sess-1', [{ content: '진행중 항목', id: '1', status: 'in_progress' }])
    })

    render(<AgentsPanelContent />)

    expect(screen.queryByText('진행중 항목')).toBeTruthy()
    expect(screen.queryAllByLabelText('Cancel this task').length).toBe(1)
    expect(screen.queryAllByLabelText('Dismiss from this panel').length).toBe(1) // row-level only

    await act(async () => {
      fireEvent.click(screen.getByLabelText('Cancel this task'))
      await Promise.resolve()
      await Promise.resolve()
    })

    // Gateway is disconnected in this test, so the optimistic cancel rolls
    // back — the item must still be present, not silently disappear.
    expect(screen.queryByText('진행중 항목')).toBeTruthy()
  })
})
