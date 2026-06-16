import { DndContext } from '@dnd-kit/core'
import { SortableContext } from '@dnd-kit/sortable'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { SessionInfo } from '@/types/hermes'

import { VirtualSessionList } from './virtual-session-list'

vi.mock('@tanstack/react-virtual', () => ({
  useVirtualizer: ({ count, getItemKey }: { count: number; getItemKey?: (index: number) => string | number }) => ({
    getTotalSize: () => count * 28,
    getVirtualItems: () =>
      Array.from({ length: count }, (_, index) => ({
        end: (index + 1) * 28,
        index,
        key: getItemKey?.(index) ?? index,
        size: 28,
        start: index * 28
      })),
    measureElement: vi.fn()
  })
}))

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
    title: 'First session',
    id: 's1',
    tool_call_count: 0,
    ...over
  } as SessionInfo
}

afterEach(() => {
  cleanup()
})

describe('VirtualSessionList drag behavior', () => {
  it('lets pointer-dnd own virtual sortable rows instead of native HTML5 drag', () => {
    const onSessionDragStart = vi.fn()
    const onPointerDown = vi.fn()

    render(
      <DndContext>
        <SortableContext items={['s1']}>
          <VirtualSessionList
            activeSessionId={null}
            onArchiveSession={vi.fn()}
            onDeleteSession={vi.fn()}
            onResumeSession={vi.fn()}
            onSessionDragStart={onSessionDragStart}
            onTogglePin={vi.fn()}
            pinned={false}
            sectionKey="sessions"
            sessionDragEnabled
            sessions={[session()]}
            sortable
            workingSessionIdSet={new Set()}
          />
        </SortableContext>
      </DndContext>
    )

    const row = screen.getByText('First session').closest('[data-session-id]') as HTMLElement

    row.addEventListener('pointerdown', onPointerDown)

    expect(row.draggable).toBe(false)
    expect(row.dataset.sessionDragSource).toBeUndefined()

    fireEvent.pointerDown(row)

    expect(onPointerDown).toHaveBeenCalledTimes(1)
    expect(onSessionDragStart).not.toHaveBeenCalled()
  })
})
