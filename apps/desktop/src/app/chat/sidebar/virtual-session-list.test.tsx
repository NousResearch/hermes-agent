import { DndContext } from '@dnd-kit/core'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { readSessionDrag } from '@/app/chat/composer/inline-refs'
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
})

describe('VirtualSessionList drag behavior', () => {
  it('keeps virtual sortable rows available for cross-section session drags', () => {
    const onSessionDragStart = vi.fn()

    render(
      <DndContext>
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
      </DndContext>
    )

    const row = screen.getByText('First session').closest('[data-session-id]') as HTMLElement
    const transfer = fakeTransfer()

    expect(row.draggable).toBe(true)
    expect(row.dataset.sessionDragSource).toBe('true')

    fireEvent.dragStart(row, { dataTransfer: transfer })

    expect(readSessionDrag(transfer)).toMatchObject({
      archived: false,
      id: 's1',
      pinId: 's1',
      pinned: false,
      profile: 'default',
      title: 'First session'
    })
    expect(onSessionDragStart).toHaveBeenCalledWith(
      expect.objectContaining({
        id: 's1',
        pinned: false
      })
    )
  })
})
