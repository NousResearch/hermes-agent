import { cleanup, fireEvent, render } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { readSessionDrag } from '@/app/chat/composer/inline-refs'
import type { SessionInfo } from '@/types/hermes'

import { SidebarSessionRow, type SidebarSessionRowProps } from './session-row'

function session(over: Partial<SessionInfo> = {}): SessionInfo {
  return {
    archived: false,
    cwd: null,
    ended_at: null,
    _lineage_root_id: null,
    input_tokens: 0,
    is_active: false,
    last_active: 1000,
    message_count: 3,
    model: null,
    output_tokens: 0,
    preview: null,
    profile: 'default',
    source: null,
    started_at: 1000,
    title: 'Test session',
    id: 's1',
    tool_call_count: 0,
    ...over
  } as SessionInfo
}

function renderRow(over: Partial<SidebarSessionRowProps> = {}) {
  const handlers = {
    onArchive: vi.fn(),
    onDelete: vi.fn(),
    onPin: vi.fn(),
    onResume: vi.fn(),
    onToggleSelect: vi.fn()
  }

  const utils = render(
    <SidebarSessionRow
      isPinned={false}
      isSelected={false}
      isWorking={false}
      onArchive={handlers.onArchive}
      onDelete={handlers.onDelete}
      onPin={handlers.onPin}
      onResume={handlers.onResume}
      onToggleSelect={handlers.onToggleSelect}
      selectable
      session={session()}
      {...over}
    />
  )

  const rowButton = utils.container.querySelector('[data-session-id] button') as HTMLButtonElement

  return { ...utils, handlers, rowButton }
}

function fakeTransfer(data: Record<string, string> = {}) {
  const store = { ...data }

  return {
    dropEffect: 'none',
    effectAllowed: 'uninitialized',
    getData: (type: string) => store[type] ?? '',
    setData: (type: string, value: string) => {
      store[type] = value
    },
    get types() {
      return Object.keys(store)
    }
  } as unknown as DataTransfer
}

afterEach(cleanup)

describe('SidebarSessionRow gestures', () => {
  it('plain click resumes when no selection is active', () => {
    const { handlers, rowButton } = renderRow()

    fireEvent.click(rowButton)

    expect(handlers.onResume).toHaveBeenCalledTimes(1)
    expect(handlers.onToggleSelect).not.toHaveBeenCalled()
  })

  it('shift-click on a selectable row requests a RANGE — it must not pin', () => {
    const { handlers, rowButton } = renderRow()

    fireEvent.click(rowButton, { shiftKey: true })

    // A cold shift-click is still a range request: the section seeds the
    // anchor from the open session so the starting row stays selected.
    expect(handlers.onToggleSelect).toHaveBeenCalledWith('range')
    expect(handlers.onPin).not.toHaveBeenCalled()
    expect(handlers.onResume).not.toHaveBeenCalled()
  })

  it('⌘-click toggles a row in and out — non-contiguous selection, never a new window', () => {
    const { handlers, rowButton } = renderRow()

    fireEvent.click(rowButton, { metaKey: true })
    expect(handlers.onToggleSelect).toHaveBeenLastCalledWith('single')

    fireEvent.click(rowButton, { metaKey: true })
    expect(handlers.onToggleSelect).toHaveBeenCalledTimes(2)
    expect(handlers.onResume).not.toHaveBeenCalled()
  })

  it('ctrl-click behaves like ⌘-click', () => {
    const { handlers, rowButton } = renderRow()

    fireEvent.click(rowButton, { ctrlKey: true })

    expect(handlers.onToggleSelect).toHaveBeenCalledWith('single')
    expect(handlers.onResume).not.toHaveBeenCalled()
  })

  it('alt-click also starts a selection', () => {
    const { handlers, rowButton } = renderRow()

    fireEvent.click(rowButton, { altKey: true })

    expect(handlers.onToggleSelect).toHaveBeenCalledWith('single')
    expect(handlers.onResume).not.toHaveBeenCalled()
  })

  it('with a selection active, plain click toggles, shift extends, ⌘ toggles', () => {
    const { handlers, rowButton } = renderRow({ selectionActive: true })

    fireEvent.click(rowButton)
    expect(handlers.onToggleSelect).toHaveBeenLastCalledWith('single')

    fireEvent.click(rowButton, { shiftKey: true })
    expect(handlers.onToggleSelect).toHaveBeenLastCalledWith('range')

    fireEvent.click(rowButton, { metaKey: true })
    expect(handlers.onToggleSelect).toHaveBeenLastCalledWith('single')

    expect(handlers.onResume).not.toHaveBeenCalled()
    expect(handlers.onPin).not.toHaveBeenCalled()
  })

  it('keeps the legacy shift-click pin only on rows with no selection wiring', () => {
    const { handlers, rowButton } = renderRow({ onToggleSelect: undefined, selectable: false })

    fireEvent.click(rowButton, { shiftKey: true })

    expect(handlers.onPin).toHaveBeenCalledTimes(1)
    expect(handlers.onResume).not.toHaveBeenCalled()
  })

  it('never pins an archived row, even without selection wiring', () => {
    const { handlers, rowButton } = renderRow({ archived: true, onToggleSelect: undefined, selectable: false })

    fireEvent.click(rowButton, { shiftKey: true })

    expect(handlers.onPin).not.toHaveBeenCalled()
    expect(handlers.onResume).not.toHaveBeenCalled()
  })

  it('renders a checked checkbox while its section is selecting', () => {
    const { container } = renderRow({ checked: true, selectionActive: true })

    const checkbox = container.querySelector('[role="checkbox"]')

    expect(checkbox).toBeTruthy()
    expect(checkbox?.getAttribute('aria-checked')).toBe('true')
  })

  it('starts a session drag from the row body without rendering a separate reorder handle', () => {
    const onSessionDragEnd = vi.fn()
    const onSessionDragStart = vi.fn()
    const { container, rowButton } = renderRow({ isPinned: true, onSessionDragEnd, onSessionDragStart, reorderable: true })
    const transfer = fakeTransfer()

    expect(container.querySelector('[data-reorder-handle]')).toBeNull()
    expect(container.querySelector('[data-drop-indicator]')).toBeNull()

    fireEvent.dragStart(rowButton, { dataTransfer: transfer })

    expect(readSessionDrag(transfer)).toMatchObject({
      archived: false,
      id: 's1',
      pinId: 's1',
      pinned: true,
      profile: 'default',
      title: 'Test session'
    })
    expect(onSessionDragStart).toHaveBeenCalledWith(
      expect.objectContaining({
        id: 's1',
        pinned: true
      })
    )

    fireEvent.dragEnd(rowButton)
    expect(onSessionDragEnd).toHaveBeenCalledTimes(1)
  })
})
