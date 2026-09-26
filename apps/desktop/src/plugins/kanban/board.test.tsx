import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest'

import { Card } from './board'
import type { KanbanTask } from './types'
import type * as KanbanUi from './ui'

beforeAll(() => {
  Element.prototype.scrollIntoView = vi.fn()
  Element.prototype.hasPointerCapture = vi.fn(() => false)
  Element.prototype.releasePointerCapture = vi.fn()
})

vi.mock('./ui', async importOriginal => {
  const actual = await importOriginal<typeof KanbanUi>()

  return {
    ...actual,
    ago: () => '',
    arcState: () => null,
    columnLabel: (_k: unknown, status: string) => (status === 'todo' ? 'Todo' : status),
    useDefaultAssignee: () => '',
    useKanban: () => ({
      delete: 'Delete',
      deselect: 'Deselect',
      moveTo: (status: string) => `Move to ${status}`,
      open: 'Open',
      select: (modifier: string) => `Select (${modifier})`
    }),
    useOrchestration: () => null
  }
})

const task: KanbanTask = {
  body: 'The card body',
  created_at: 1,
  id: 't_fixture_123',
  status: 'todo',
  title: 'Keyboard accessible task'
}

function renderCard() {
  const callbacks = {
    onDelete: vi.fn(),
    onMove: vi.fn(),
    onOpen: vi.fn(),
    onToggleSelect: vi.fn()
  }

  render(<Card columns={['todo', 'done']} selected={false} task={task} {...callbacks} />)

  const card = screen.getByText(task.title).closest<HTMLElement>('[draggable="true"]')

  if (!card) {
    throw new Error('Task card root was not rendered')
  }

  return { card, ...callbacks }
}

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

describe('Kanban task card interactions', () => {
  it('opens exactly once from a pointer click', async () => {
    const user = userEvent.setup()
    const { card, onOpen, onToggleSelect } = renderCard()

    await user.click(card)

    expect(onOpen).toHaveBeenCalledOnce()
    expect(onOpen).toHaveBeenCalledWith(task.id)
    expect(onToggleSelect).not.toHaveBeenCalled()
  })

  it.each(['metaKey', 'ctrlKey'] as const)('toggles selection without opening for %s-click', modifier => {
    const { card, onOpen, onToggleSelect } = renderCard()

    fireEvent.click(card, { [modifier]: true })

    expect(onToggleSelect).toHaveBeenCalledOnce()
    expect(onToggleSelect).toHaveBeenCalledWith(task.id)
    expect(onOpen).not.toHaveBeenCalled()
  })

  it('is a named native button in the normal Tab order', async () => {
    const user = userEvent.setup()
    renderCard()

    const card = screen.getByRole('button', {
      name: /Keyboard accessible task.*t_fixture_123.*Todo/i
    })

    expect(card.getAttribute('type')).toBe('button')
    expect(card.classList.contains('kanban-task-card')).toBe(true)

    await user.tab()

    expect(card.ownerDocument.activeElement).toBe(card)
  })

  it.each(['{Enter}', ' '])('opens exactly once from %s', async key => {
    const user = userEvent.setup()
    const { onOpen } = renderCard()
    const card = screen.getByRole('button', { name: /Keyboard accessible task/i })
    card.focus()

    await user.keyboard(key)

    expect(onOpen).toHaveBeenCalledOnce()
    expect(onOpen).toHaveBeenCalledWith(task.id)
  })

  it('preserves the native drag payload contract', () => {
    const { card } = renderCard()

    const dataTransfer = {
      effectAllowed: 'none',
      setData: vi.fn(),
      setDragImage: vi.fn()
    }

    expect(card.getAttribute('draggable')).toBe('true')

    fireEvent.dragStart(card, { dataTransfer })

    expect(dataTransfer.setData).toHaveBeenCalledWith('text/plain', task.id)
    expect(dataTransfer.effectAllowed).toBe('move')
    expect(dataTransfer.setDragImage).toHaveBeenCalledOnce()
  })

  it('keeps context-menu selection isolated from card activation', async () => {
    const user = userEvent.setup()
    const { card, onOpen, onToggleSelect } = renderCard()

    fireEvent.pointerDown(card, { button: 2, pointerType: 'mouse' })
    fireEvent.contextMenu(card, { button: 2 })

    expect(onOpen).not.toHaveBeenCalled()
    await user.click(await screen.findByRole('menuitem', { name: /Select/i }))

    expect(onToggleSelect).toHaveBeenCalledOnce()
    expect(onOpen).not.toHaveBeenCalled()
  })

  it('contains no nested interactive controls', () => {
    renderCard()
    const card = screen.getByRole('button', { name: /Keyboard accessible task/i })

    expect(card.querySelector('button, a[href], input, select, textarea, [role="button"], [tabindex]')).toBeNull()
  })
})
