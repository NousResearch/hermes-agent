import { cleanup, createEvent, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { readSessionDrag } from '@/app/chat/composer/inline-refs'
import { $attentionSessionIds } from '@/store/session'
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
    onHaltSelectedSessions: vi.fn(),
    onPin: vi.fn(),
    onPromptSelectedSessions: vi.fn(),
    onResume: vi.fn(),
    onSteerSelectedSessions: vi.fn(),
    onToggleSelect: vi.fn()
  }

  const utils = render(
    <SidebarSessionRow
      isPinned={false}
      isSelected={false}
      isWorking={false}
      onArchive={handlers.onArchive}
      onDelete={handlers.onDelete}
      onHaltSelectedSessions={handlers.onHaltSelectedSessions}
      onPin={handlers.onPin}
      onPromptSelectedSessions={handlers.onPromptSelectedSessions}
      onResume={handlers.onResume}
      onSteerSelectedSessions={handlers.onSteerSelectedSessions}
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

afterEach(() => {
  cleanup()
  $attentionSessionIds.set([])
})

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

  it('right-clicking a checked multi-selected row opens bulk actions for the selected set', async () => {
    const onArchiveSelectedSessions = vi.fn()
    const onHaltSelectedSessions = vi.fn()

    const { handlers, rowButton } = renderRow({
      bulkSelectedSessionIds: ['s1', 's2', 's3'],
      checked: true,
      onArchiveSelectedSessions,
      onHaltSelectedSessions,
      selectionActive: true
    })

    fireEvent.contextMenu(rowButton)

    expect(await screen.findByText('Prompt 3')).toBeTruthy()
    expect(screen.getByText('Steer 3')).toBeTruthy()
    expect(screen.getByText('Stop 3')).toBeTruthy()
    expect(screen.getByText('Archive 3')).toBeTruthy()
    expect(screen.getByText('Delete 3')).toBeTruthy()
    expect(screen.queryByText('Rename')).toBeNull()

    fireEvent.click(screen.getByText('Stop 3'))

    await waitFor(() => expect(onHaltSelectedSessions).toHaveBeenCalledWith(['s1', 's2', 's3']))
    expect(handlers.onDelete).not.toHaveBeenCalled()
  })

  it('right-clicking a checked multi-selected row archives the selected set', async () => {
    const onArchiveSelectedSessions = vi.fn()

    const { handlers, rowButton } = renderRow({
      bulkSelectedSessionIds: ['s1', 's2', 's3'],
      checked: true,
      onArchiveSelectedSessions,
      selectionActive: true
    })

    fireEvent.contextMenu(rowButton)

    expect(await screen.findByText('Archive 3')).toBeTruthy()
    expect(screen.getByText('Delete 3')).toBeTruthy()
    expect(screen.queryByText('Rename')).toBeNull()

    fireEvent.click(screen.getByText('Archive 3'))

    await waitFor(() => expect(onArchiveSelectedSessions).toHaveBeenCalledWith(['s1', 's2', 's3']))
    expect(handlers.onArchive).not.toHaveBeenCalled()
  })

  it('keeps destructive session deletes adjacent with standard Delete last', async () => {
    const { rowButton } = renderRow()

    fireEvent.contextMenu(rowButton)

    expect(await screen.findByText('Delete cloud channel')).toBeTruthy()

    const labels = screen
      .getAllByRole('menuitem')
      .map(item => item.textContent?.trim())
      .filter(Boolean)

    expect(labels.slice(-3)).toEqual(['Archive', 'Delete cloud channel', 'Delete'])
  })

  it('moves the timestamp for a real actions menu open, not lingering pointer focus', () => {
    const { container } = renderRow()
    const chrome = container.querySelector('[data-session-row-chrome]') as HTMLElement
    const trailing = container.querySelector('[data-session-row-trailing]') as HTMLElement
    const timestamp = container.querySelector('[data-session-row-age]') as HTMLElement
    const actionsButton = container.querySelector('[data-session-row-actions]') as HTMLButtonElement

    expect(trailing.className).toContain('justify-end')
    expect(trailing.className).not.toContain('pr-7')
    expect(timestamp.className).toContain('pr-1')
    expect(timestamp.className).toContain('group-data-[actions-visible=true]/session-row:-translate-x-6')
    expect(timestamp.className).not.toContain('group-hover:-translate-x-6')
    expect(timestamp.className).not.toContain('group-focus-within')
    expect(actionsButton.className).toContain('group-hover/session-row:opacity-100')
    expect(actionsButton.className).not.toContain('group-hover:opacity-100')
    expect(actionsButton.className).not.toContain('group-focus-within')
    expect(chrome.getAttribute('data-actions-visible')).toBeNull()

    fireEvent.pointerDown(actionsButton, { button: 0, ctrlKey: false })
    expect(chrome.getAttribute('data-actions-visible')).toBe('true')

    fireEvent.pointerDown(actionsButton, { button: 0, ctrlKey: false })
    expect(chrome.getAttribute('data-actions-visible')).toBeNull()
  })

  it('hides active timestamps but keeps waiting-user timestamps visible', () => {
    const active = renderRow({ isWorking: true })
    let timestamp = active.container.querySelector('[data-session-row-age]') as HTMLElement

    expect(timestamp.className).toContain('opacity-0')

    cleanup()
    $attentionSessionIds.set(['s1'])

    const waiting = renderRow({ isWorking: true })
    timestamp = waiting.container.querySelector('[data-session-row-age]') as HTMLElement

    expect(timestamp.className).not.toContain('opacity-0')
    expect(waiting.container.querySelector('[aria-label="Needs your input"]')).toBeTruthy()
  })

  it('uses active title contrast for selected and running rows without styling waiting input as live motion', () => {
    const selected = renderRow({ isSelected: true })
    let title = selected.container.querySelector('[data-session-row-title]') as HTMLElement

    expect(title.className).toContain('text-(--ui-text-primary)')
    expect(title.className).toContain('font-medium')

    cleanup()

    const running = renderRow({ isWorking: true })
    title = running.container.querySelector('[data-session-row-title]') as HTMLElement

    expect(title.className).toContain('text-(--ui-text-primary)')
    expect(title.className).toContain('font-medium')

    cleanup()
    $attentionSessionIds.set(['s1'])

    const waiting = renderRow({ isWorking: true })
    title = waiting.container.querySelector('[data-session-row-title]') as HTMLElement

    expect(title.className).toContain('text-(--ui-text-secondary)')
    expect(title.className).toContain('font-normal')
  })

  it('starts a session drag from the row body without rendering a separate reorder handle', () => {
    const onSessionDragEnd = vi.fn()
    const onSessionDragStart = vi.fn()

    const { container, rowButton } = renderRow({
      isPinned: true,
      onSessionDragEnd,
      onSessionDragStart,
      reorderable: true
    })

    const transfer = fakeTransfer()
    const dragAnchor = container.querySelector('[data-session-id]') as HTMLElement
    const dragSource = container.querySelector('[data-session-row-chrome]') as HTMLElement
    const timestamp = container.querySelector('[data-session-row-age]') as HTMLElement

    expect(container.querySelector('[data-reorder-handle]')).toBeNull()
    expect(container.querySelector('[data-drop-indicator]')).toBeNull()
    expect(dragAnchor.draggable).toBe(false)
    expect(dragSource.dataset.sessionDragSource).toBe('true')
    expect(dragSource.draggable).toBe(true)
    expect(dragSource.className).toContain('[-webkit-app-region:no-drag]')
    expect(rowButton.draggable).toBe(false)

    fireEvent.dragStart(dragSource, { dataTransfer: transfer })

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
    expect(onSessionDragStart).toHaveBeenCalledTimes(1)

    fireEvent.dragEnd(dragSource)
    expect(onSessionDragEnd).toHaveBeenCalledTimes(1)

    const timestampTransfer = fakeTransfer()

    fireEvent.dragStart(timestamp, { dataTransfer: timestampTransfer })

    expect(readSessionDrag(timestampTransfer)).toMatchObject({
      id: 's1',
      pinned: true
    })
    expect(onSessionDragStart).toHaveBeenCalledTimes(2)
  })

  it('does not turn the row actions menu into a session drag source', () => {
    const onSessionDragStart = vi.fn()
    const { container } = renderRow({ onSessionDragStart, reorderable: true })
    const actionsButton = container.querySelector('[data-session-row-actions]') as HTMLButtonElement
    const transfer = fakeTransfer()
    const event = createEvent.dragStart(actionsButton, { dataTransfer: transfer })

    const notCanceled = fireEvent(actionsButton, event)

    expect(notCanceled).toBe(false)
    expect(event.defaultPrevented).toBe(true)
    expect(readSessionDrag(transfer)).toBeNull()
    expect(onSessionDragStart).not.toHaveBeenCalled()
  })
})
