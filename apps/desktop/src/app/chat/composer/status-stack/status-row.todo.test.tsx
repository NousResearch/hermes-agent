import React, { act } from 'react'
import { createRoot } from 'react-dom/client'
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest'

vi.mock('@/app/right-sidebar/terminal/terminals', () => ({ openAgentTerminal: vi.fn() }))
vi.mock('@/components/ui/button', () => ({
  Button: (props: React.ButtonHTMLAttributes<HTMLButtonElement>) => <button {...props} />
}))
vi.mock('@/components/ui/codicon', () => ({ Codicon: ({ name }: { name: string }) => <span data-icon={name} /> }))
vi.mock('@/components/ui/glyph-spinner', () => ({
  GlyphSpinner: ({ ariaLabel }: { ariaLabel: string }) => <span aria-label={ariaLabel} />
}))
vi.mock('@/components/ui/tooltip', () => ({ Tip: ({ children }: { children: React.ReactNode }) => <>{children}</> }))
vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      statusStack: {
        dismiss: 'Dismiss',
        exit: (code: number) => `exit ${code}`,
        markDone: 'Mark done',
        markDoneAria: (task: string) => `Mark task done: ${task}`,
        reopen: 'Reopen',
        reopenAria: (task: string) => `Reopen task: ${task}`,
        running: 'Running',
        statusCancelled: 'Cancelled',
        statusCompleted: 'Completed',
        statusInProgress: 'In progress',
        statusPending: 'Pending',
        stop: 'Stop',
        syncingTask: 'Syncing task status'
      }
    }
  })
}))
vi.mock('@/lib/text', () => ({ capitalize: (value: string) => value }))
vi.mock('@/lib/utils', () => ({ cn: (...values: unknown[]) => values.filter(Boolean).join(' ') }))

const { StatusItemRow } = await import('./status-row')

beforeAll(() => {
  ;(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true
})

afterEach(() => {
  globalThis.document.body.innerHTML = ''
})

async function renderRow(status: 'pending' | 'in_progress' | 'completed' | 'cancelled', enabled = true) {
  const container = globalThis.document.createElement('div')
  globalThis.document.body.append(container)
  const root = createRoot(container)
  const onTodoAction = vi.fn()
  await act(async () => {
    root.render(
      <StatusItemRow
        item={{
          id: 'todo:raw-id',
          state: status === 'in_progress' ? 'running' : 'done',
          title: 'Build tray',
          todoItemId: 'raw-id',
          todoStatus: status,
          type: 'todo'
        }}
        onTodoAction={onTodoAction}
        todoMutationEnabled={enabled}
      />
    )
  })

  return { button: container.querySelector('button')!, container, onTodoAction, root }
}

describe('todo status row actions', () => {
  it('marks open work done using the exact raw backend id', async () => {
    const view = await renderRow('in_progress')
    expect(view.button.getAttribute('aria-label')).toBe('Mark task done: Build tray')
    expect(view.button.parentElement?.className).not.toContain('opacity-0')
    await act(async () => view.button.click())
    expect(view.onTodoAction).toHaveBeenCalledWith(expect.objectContaining({ todoItemId: 'raw-id' }), view.button)
    await act(async () => view.root.unmount())
  })

  it('reopens terminal work', async () => {
    const view = await renderRow('completed')
    expect(view.button.getAttribute('aria-label')).toBe('Reopen task: Build tray')
    await act(async () => view.root.unmount())
  })

  it('keeps legacy display rows readable but disables mutation', async () => {
    const view = await renderRow('pending', false)
    expect(view.button.disabled).toBe(true)
    expect(view.button.getAttribute('aria-label')).toBe('Syncing task status')
    await act(async () => view.root.unmount())
  })
})
