import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { Codicon } from '@/components/ui/codicon'

import { StatusRow } from './status-row'

vi.stubGlobal(
  'ResizeObserver',
  class {
    observe() {}
    unobserve() {}
    disconnect() {}
  }
)

afterEach(cleanup)

it('keeps dismiss and nested controls independent from row activation', () => {
  const activate = vi.fn()
  const dismiss = vi.fn()
  const action = vi.fn()

  const { container } = render(
    <div data-slot="composer-status-stack">
      <StatusRow
        dismiss={{ label: 'Dismiss task', onDismiss: dismiss }}
        leading={<Codicon name="comment" />}
        onActivate={activate}
        trailing={
          <button
            onClick={event => {
              event.stopPropagation()
              action()
            }}
          >
            Edit task
          </button>
        }
      >
        <span>Task title</span>
      </StatusRow>
    </div>
  )

  const row = container.querySelector<HTMLElement>('[data-slot="status-row"]')!
  const close = screen.getByRole('button', { name: 'Dismiss task' })
  fireEvent.keyDown(close, { key: 'Enter' })
  fireEvent.click(close)
  expect(dismiss).toHaveBeenCalledOnce()
  expect(activate).not.toHaveBeenCalled()
  fireEvent.keyDown(screen.getByRole('button', { name: 'Edit task' }), { key: ' ' })
  fireEvent.click(screen.getByRole('button', { name: 'Edit task' }))
  expect(action).toHaveBeenCalledOnce()
  expect(activate).not.toHaveBeenCalled()
  fireEvent.keyDown(row, { key: 'Enter' })
  fireEvent.keyDown(row, { key: ' ' })
  expect(activate).toHaveBeenCalledTimes(2)
})
