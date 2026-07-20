import { act, cleanup, fireEvent, render, screen, within } from '@testing-library/react'
import { atom } from 'nanostores'
import type { ReactNode } from 'react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'
import type { TodoHistorySnapshot, TodoItem } from '@/lib/todos'
import { $messages } from '@/store/session'
import {
  $todoHistoryBySession,
  clearAllSessionTodoState,
  rebuildSessionTodoHistory,
  setSessionTodos
} from '@/store/todos'

import { ComposerStatusStack } from '.'

class ResizeObserverStub {
  disconnect() {}
  observe() {}
}

const todo = (id: string, content: string, status: TodoItem['status'] = 'completed'): TodoItem => ({
  content,
  id,
  status
})

const snapshot = (id: string, content: string): TodoHistorySnapshot => ({
  id,
  state: 'completed',
  todos: [todo(id, content)]
})

function renderStack(sessionId: string, wrapper?: (children: ReactNode) => ReactNode) {
  const stack = <ComposerStatusStack queue={null} sessionId={sessionId} />

  return render(
    <MemoryRouter>
      <I18nProvider configClient={null} initialLocale="en">
        {wrapper ? wrapper(stack) : stack}
      </I18nProvider>
    </MemoryRouter>
  )
}

describe('composer task history', () => {
  beforeAll(() => {
    vi.stubGlobal('ResizeObserver', ResizeObserverStub)
  })

  afterEach(() => {
    cleanup()
    clearAllSessionTodoState()
    vi.useRealTimers()
    vi.restoreAllMocks()
  })

  it('keeps task history available after the finished live list dismisses at four seconds', () => {
    vi.useFakeTimers()
    $todoHistoryBySession.set({ sid: [snapshot('old', 'Historical task')] })
    setSessionTodos('sid', [todo('live', 'Live task')])

    const view = renderStack('sid')

    expect(screen.getByText('Live task')).toBeTruthy()
    const historyButton = screen.getByRole('button', { name: 'Task history' })
    expect(historyButton.getAttribute('aria-expanded')).toBe('false')
    const controlledId = historyButton.getAttribute('aria-controls') ?? ''
    expect(controlledId).toBeTruthy()

    act(() => vi.advanceTimersByTime(4_000))

    expect(screen.queryByText('Live task')).toBeNull()
    expect(screen.getByRole('button', { name: 'Task history' })).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: 'Task history' }))
    expect(screen.getByText('Historical task')).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Task history' }).getAttribute('aria-expanded')).toBe('true')
    expect(view.container.querySelector(`[id="${controlledId}"]`)).toBeTruthy()
  })

  it('renders the live list first and keeps a large history in a separate collapsed section', () => {
    const history = Array.from({ length: 10 }, (_, index) => snapshot(`history-${index}`, `History ${index}`))
    $todoHistoryBySession.set({ sid: history })
    setSessionTodos('sid', [todo('live', 'Live now', 'in_progress')])

    const view = renderStack('sid')
    const live = screen.getByText('Live now')
    const historyButton = screen.getByRole('button', { name: 'Task history' })

    expect(live).toBeTruthy()
    expect(historyButton.getAttribute('aria-expanded')).toBe('false')
    expect(screen.queryByText('History 0')).toBeNull()
    expect(live.compareDocumentPosition(historyButton) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
    expect(view.container.querySelectorAll('button[aria-expanded]')).toHaveLength(2)
  })

  it('shows only the newest copy of the same id and content when status changed', () => {
    rebuildSessionTodoHistory('sid', [
      {
        id: 'older',
        role: 'assistant',
        parts: [
          {
            type: 'tool-call',
            toolName: 'todo',
            toolCallId: 'todo-old',
            args: { todos: [todo('same', 'One plan', 'in_progress')] }
          }
        ]
      },
      {
        id: 'newer',
        role: 'assistant',
        parts: [
          {
            type: 'tool-call',
            toolName: 'todo',
            toolCallId: 'todo-new',
            result: { todos: [todo('same', 'One plan', 'completed')] }
          }
        ]
      }
    ])

    renderStack('sid')
    fireEvent.click(screen.getByRole('button', { name: 'Task history' }))

    expect(screen.getAllByText('One plan')).toHaveLength(1)
  })

  it('isolates task history between two session views', () => {
    $todoHistoryBySession.set({
      'runtime-a': [snapshot('a', 'Only A')],
      'runtime-b': [snapshot('b', 'Only B')]
    })

    const view = render(
      <MemoryRouter>
        <I18nProvider configClient={null} initialLocale="en">
          <section aria-label="Session A">
            <ComposerStatusStack queue={null} sessionId="runtime-a" />
          </section>
          <section aria-label="Session B">
            <ComposerStatusStack queue={null} sessionId="runtime-b" />
          </section>
        </I18nProvider>
      </MemoryRouter>
    )

    const sessionA = within(view.getByRole('region', { name: 'Session A' }))
    const sessionB = within(view.getByRole('region', { name: 'Session B' }))
    fireEvent.click(sessionA.getByRole('button', { name: 'Task history' }))

    expect(sessionA.getByText('Only A')).toBeTruthy()
    expect(sessionA.queryByText('Only B')).toBeNull()
    expect(sessionB.queryByText('Only A')).toBeNull()
    expect(sessionB.queryByText('Only B')).toBeNull()

    fireEvent.click(sessionB.getByRole('button', { name: 'Task history' }))
    expect(sessionB.getByText('Only B')).toBeTruthy()
  })

  it('does not subscribe the composer status surface to transcript messages', () => {
    const listen = vi.spyOn($messages, 'listen')
    const unrelatedMessages = atom([{ id: 'message', role: 'user' }])

    renderStack('sid', children => <div data-unrelated-message-count={unrelatedMessages.get().length}>{children}</div>)

    expect(listen).not.toHaveBeenCalled()
  })
})
