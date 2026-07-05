import { cleanup, fireEvent, render } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { readSessionDrag } from '@/app/chat/composer/inline-refs'
import type { SessionInfo } from '@/hermes'
import { $attentionSessionIds } from '@/store/session'

import { SidebarSessionRow } from './session-row'

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

function renderRow() {
  const dragHandleProps = {
    onKeyDown: vi.fn(),
    onMouseDown: vi.fn(),
    onPointerDown: vi.fn(),
    onTouchStart: vi.fn()
  }

  const utils = render(
    <SidebarSessionRow
      dragHandleProps={dragHandleProps}
      isPinned={false}
      isSelected={false}
      isWorking={false}
      onArchive={vi.fn()}
      onDelete={vi.fn()}
      onPin={vi.fn()}
      onResume={vi.fn()}
      reorderable
      session={session()}
    />
  )

  return { ...utils, dragHandleProps }
}

function renderNativeRow() {
  return render(
    <SidebarSessionRow
      isPinned={false}
      isSelected={false}
      isWorking={false}
      onArchive={vi.fn()}
      onDelete={vi.fn()}
      onPin={vi.fn()}
      onResume={vi.fn()}
      session={session()}
    />
  )
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

describe('SidebarSessionRow reorder activation', () => {
  it('starts reorder from row chrome, not just the tiny grab handle', () => {
    const { container, dragHandleProps } = renderRow()
    const chrome = container.querySelector('[data-session-row-chrome]') as HTMLElement

    fireEvent.mouseDown(chrome)

    expect(dragHandleProps.onMouseDown).toHaveBeenCalledTimes(1)
  })

  it('keeps the main row button wired without double-firing through row chrome', () => {
    const { container, dragHandleProps } = renderRow()
    const main = container.querySelector('[data-session-row-main]') as HTMLElement

    fireEvent.mouseDown(main)

    expect(dragHandleProps.onMouseDown).toHaveBeenCalledTimes(1)
  })

  it('does not start reorder from the actions menu button', () => {
    const { container, dragHandleProps } = renderRow()
    const actions = container.querySelector('[data-session-row-actions]') as HTMLElement

    fireEvent.mouseDown(actions)

    expect(dragHandleProps.onMouseDown).not.toHaveBeenCalled()
  })
})

describe('SidebarSessionRow native drag activation', () => {
  it('shows native session drags as active until drag end', () => {
    const { container } = renderNativeRow()
    const chrome = container.querySelector('[data-session-row-chrome]') as HTMLElement
    const transfer = fakeTransfer()

    fireEvent.dragStart(chrome, { dataTransfer: transfer })

    expect(readSessionDrag(transfer)).toMatchObject({
      id: 's1',
      profile: 'default',
      title: 'Test session'
    })
    expect(chrome.className).toContain('cursor-grabbing')

    fireEvent.dragEnd(chrome)

    expect(chrome.className).not.toContain('cursor-grabbing')
  })
})
