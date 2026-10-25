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

afterEach(() => {
  cleanup()
  window.getSelection()?.removeAllRanges()
})

it('allows copying row text without activating it and preserves deliberate activation', () => {
  const activate = vi.fn()

  const { container } = render(
    <div data-slot="composer-status-stack">
      <StatusRow leading={<span data-testid="row-icon">↗</span>} onActivate={activate}>
        <span>Task objective</span>
      </StatusRow>
      <p>Unrelated selection</p>
    </div>
  )

  const text = screen.getByText('Task objective')
  const row = text.closest('[data-slot="status-row"]')!
  const icon = screen.getByTestId('row-icon')
  const content = container.querySelector('.status-row-content')!
  const selection = window.getSelection()!
  const range = text.ownerDocument.createRange()
  fireEvent.pointerDown(text, { clientX: 10, clientY: 10 })
  fireEvent.mouseDown(text, { clientX: 10, clientY: 10 })
  fireEvent.pointerMove(text, { clientX: 80, clientY: 10 })
  fireEvent.mouseMove(text, { clientX: 80, clientY: 10 })
  range.selectNodeContents(text)
  selection.addRange(range)
  fireEvent.pointerUp(text, { clientX: 80, clientY: 10 })
  fireEvent.mouseUp(text, { clientX: 80, clientY: 10 })
  fireEvent.click(text, { detail: 1, clientX: 80, clientY: 10 })
  expect(activate).not.toHaveBeenCalled()
  expect(getComputedStyle(content).userSelect).toBe('text')
  expect(selection.toString()).toBe(text.textContent)

  fireEvent.keyDown(row, { key: 'Enter' })
  fireEvent.keyDown(row, { key: ' ' })
  fireEvent.click(row, { detail: 0 })
  expect(activate).toHaveBeenCalledTimes(3)

  // The browser can leave this selection in place after a later ordinary click.
  fireEvent.pointerDown(icon, { clientX: 5, clientY: 5 })
  fireEvent.mouseDown(icon, { clientX: 5, clientY: 5 })
  fireEvent.pointerUp(icon, { clientX: 5, clientY: 5 })
  fireEvent.mouseUp(icon, { clientX: 5, clientY: 5 })
  fireEvent.click(icon, { detail: 1, clientX: 5, clientY: 5 })
  expect(activate).toHaveBeenCalledTimes(4)

  fireEvent.mouseDown(text, { clientX: 10, clientY: 10 })
  fireEvent.mouseUp(text, { clientX: 10, clientY: 10 })
  fireEvent.click(text, { detail: 1, clientX: 10, clientY: 10 })
  expect(activate).toHaveBeenCalledTimes(5)

  selection.removeAllRanges()
  range.selectNodeContents(screen.getByText('Unrelated selection'))
  selection.addRange(range)
  fireEvent.click(text, { detail: 1 })
  expect(activate).toHaveBeenCalledTimes(6)
})

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
