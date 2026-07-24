import { act, cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import type { TodoItem } from '@/lib/todos'
import { setActiveSessionId } from '@/store/session'
import { clearSessionTodos, setSessionTodos } from '@/store/todos'

import { deriveRunBoardState, RunBoardPane } from './run-board'

const todo = (id: string, content: string, status: TodoItem['status']): TodoItem => ({ content, id, status })

afterEach(() => {
  cleanup()
  setActiveSessionId(null)
  clearSessionTodos('session-a')
  clearSessionTodos('session-b')
})

describe('deriveRunBoardState', () => {
  it('reports the active gate and completed progress', () => {
    const state = deriveRunBoardState([
      todo('done', 'Discovery', 'completed'),
      todo('active', 'Build the persistent pane', 'in_progress'),
      todo('next', 'Verify the pane', 'pending')
    ])

    expect(state.kind).toBe('active')
    expect(state.current?.id).toBe('active')
    expect(state.completed).toBe(1)
    expect(state.total).toBe(3)
    expect(state.supportNeeded).toBe(false)
  })

  it('recognizes an explicit needs-you blocker', () => {
    const state = deriveRunBoardState([
      todo('blocked', 'BLOCKED: needs you — approve the production restart', 'in_progress')
    ])

    expect(state.kind).toBe('blockedNeedsYou')
    expect(state.supportNeeded).toBe(true)
  })

  it('reports done only when every gate is resolved', () => {
    expect(
      deriveRunBoardState([todo('done', 'Shipped', 'completed'), todo('skip', 'Deferred', 'cancelled')]).kind
    ).toBe('done')
  })
})

describe('RunBoardPane', () => {
  it('shows only the active runtime session plan', () => {
    setSessionTodos('session-a', [todo('a', 'Build session A', 'in_progress')])
    setSessionTodos('session-b', [todo('b', 'Do not show session B', 'in_progress')])
    setActiveSessionId('session-a')

    render(<RunBoardPane />)

    expect(screen.getByText('Run board')).toBeTruthy()
    expect(screen.getByText('ACTIVE')).toBeTruthy()
    expect(screen.getByText('Build session A')).toBeTruthy()
    expect(screen.queryByText('Do not show session B')).toBeNull()
    expect(screen.getByText('Support needed: No')).toBeTruthy()
  })

  it('follows session switches without mixing plans', () => {
    setSessionTodos('session-a', [todo('a', 'Session A gate', 'in_progress')])
    setSessionTodos('session-b', [todo('b', 'Session B gate', 'pending')])
    setActiveSessionId('session-a')
    render(<RunBoardPane />)

    act(() => setActiveSessionId('session-b'))

    expect(screen.queryByText('Session A gate')).toBeNull()
    expect(screen.getByText('Session B gate')).toBeTruthy()
  })

  it('keeps an always-present empty surface before a plan exists', () => {
    setActiveSessionId('session-a')

    render(<RunBoardPane />)

    expect(screen.getByText('No task plan yet')).toBeTruthy()
  })
})
