import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { TodoItem } from '@/lib/todos'
import { setActiveSessionId, setBusy } from '@/store/session'
import { $runBoardRefresh, clearSessionTodos, setSessionTodos } from '@/store/todos'

import { deriveRunBoardState, RunBoardPane } from './run-board'

const todo = (id: string, content: string, status: TodoItem['status']): TodoItem => ({ content, id, status })

afterEach(() => {
  cleanup()
  setActiveSessionId(null)
  setBusy(false)
  clearSessionTodos('session-a')
  clearSessionTodos('session-b')
  $runBoardRefresh.set(null)
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

  it('exposes a manual refresh control in the board header', () => {
    setActiveSessionId('session-a')

    render(<RunBoardPane />)

    expect(screen.getByRole('button', { name: 'Refresh run board' })).toBeTruthy()
  })

  it('coalesces repeated refresh clicks while reconciliation is pending', async () => {
    let finish!: () => void

    const refresh = vi.fn(
      () =>
        new Promise<void>(resolve => {
          finish = resolve
        })
    )

    $runBoardRefresh.set(refresh)
    setActiveSessionId('session-a')
    render(<RunBoardPane />)
    const button = screen.getByRole('button', { name: 'Refresh run board' })

    fireEvent.click(button)
    fireEvent.click(button)

    expect(refresh).toHaveBeenCalledTimes(1)
    expect((button as HTMLButtonElement).disabled).toBe(true)
    expect(screen.getByText('Refreshing…')).toBeTruthy()

    finish()
    await screen.findByText('Refreshed')
    expect((button as HTMLButtonElement).disabled).toBe(false)
  })

  it('does not refresh persisted state while the active turn is busy', () => {
    const refresh = vi.fn(() => Promise.resolve())

    $runBoardRefresh.set(refresh)
    setActiveSessionId('session-a')
    setBusy(true)
    render(<RunBoardPane />)
    const button = screen.getByRole('button', { name: 'Refresh run board' })

    expect((button as HTMLButtonElement).disabled).toBe(true)
    fireEvent.click(button)
    expect(refresh).not.toHaveBeenCalled()
  })

  it('surfaces refresh failure without discarding the current plan', async () => {
    $runBoardRefresh.set(() => Promise.reject(new Error('gateway offline')))
    setSessionTodos('session-a', [todo('active', 'Preserve interrupted work', 'in_progress')])
    setActiveSessionId('session-a')
    render(<RunBoardPane />)
    const button = screen.getByRole('button', { name: 'Refresh run board' })

    fireEvent.click(button)

    await screen.findByText('Refresh failed')
    expect(screen.getByText('Preserve interrupted work')).toBeTruthy()
    expect((button as HTMLButtonElement).disabled).toBe(false)
  })

  it('clears feedback when the displayed session changes', async () => {
    $runBoardRefresh.set(() => Promise.resolve())
    setActiveSessionId('session-a')
    render(<RunBoardPane />)

    fireEvent.click(screen.getByRole('button', { name: 'Refresh run board' }))
    await screen.findByText('Refreshed')

    act(() => setActiveSessionId('session-b'))

    expect(screen.queryByText('Refreshed')).toBeNull()
  })

  it('lets the displayed session refresh while another session read is pending', async () => {
    let finishFirst!: () => void

    const refresh = vi
      .fn()
      .mockImplementationOnce(
        () =>
          new Promise<void>(resolve => {
            finishFirst = resolve
          })
      )
      .mockResolvedValueOnce(undefined)

    $runBoardRefresh.set(refresh)
    setActiveSessionId('session-a')
    render(<RunBoardPane />)
    fireEvent.click(screen.getByRole('button', { name: 'Refresh run board' }))

    act(() => setActiveSessionId('session-b'))

    const secondButton = screen.getByRole('button', { name: 'Refresh run board' })

    expect((secondButton as HTMLButtonElement).disabled).toBe(false)
    fireEvent.click(secondButton)
    await screen.findByText('Refreshed')
    expect(refresh).toHaveBeenCalledTimes(2)

    await act(async () => {
      finishFirst()
      await Promise.resolve()
    })
    expect(screen.getByText('Refreshed')).toBeTruthy()
  })
})
