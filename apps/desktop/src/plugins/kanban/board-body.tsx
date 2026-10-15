/** The board's body below the header: the empty state, and the columns —
 *  the swimlane grid when a grouping is on, else the flat strip of columns.
 *  Owns the grab-to-scrub scroller for both layouts. */

import { Button, cn, Codicon, useGrabScroll } from '@hermes/plugin-sdk'
import { useRef } from 'react'

import { Column } from './column'
import { SwimlaneGrid } from './swimlane-grid'
import type { KanbanColumn } from './types'
import { useKanban } from './ui'
import type { NewTaskRequest, useSwimlanes } from './use-swimlanes'

export function EmptyBoard({ filtered, onNewTask }: { filtered: boolean; onNewTask: () => void }) {
  const k = useKanban()

  return (
    <div className="grid flex-1 place-items-center px-4 text-center">
      <div className="flex flex-col items-center gap-2">
        <Codicon className="text-(--ui-text-quaternary)" name="project" size="1.25rem" />
        <p className="text-xs text-(--ui-text-tertiary)">{filtered ? k.noMatch : k.noTasks}</p>
        <Button className="mt-0.5" onClick={onNewTask} size="sm" variant="outline">
          <Codicon name="add" size="0.75rem" />
          {k.newTask}
        </Button>
      </div>
    </div>
  )
}

export interface CardActions {
  onDelete: (id: string) => void
  onMove: (id: string, status: string) => void
  onOpen: (id: string) => void
  onToggleSelect: (id: string) => void
  selected: ReadonlySet<string>
}

export interface ColumnCollapse {
  isCollapsed: (column: KanbanColumn) => boolean
  toggle: (column: KanbanColumn) => void
}

export function BoardColumns({
  cards,
  collapse,
  columns,
  onAdd,
  swim
}: {
  cards: CardActions
  collapse: ColumnCollapse
  columns: KanbanColumn[]
  onAdd: (request: NewTaskRequest) => void
  swim: ReturnType<typeof useSwimlanes>
}) {
  // Grab-to-scrub the lane strip (shared primitive, same as the dashboard's pan).
  const scrollRef = useRef<HTMLDivElement>(null)
  const { grabbing, onMouseDown } = useGrabScroll(scrollRef)
  const names = columns.map(col => col.name)

  if (swim.by && swim.lanes) {
    return (
      <div
        className={cn('min-h-0 flex-1 overflow-auto px-4 pb-3', grabbing && 'cursor-grabbing')}
        onMouseDown={onMouseDown}
        ref={scrollRef}
      >
        <SwimlaneGrid
          {...cards}
          by={swim.by}
          canDrop={swim.canDrop}
          collapsedColumn={collapse.isCollapsed}
          columns={columns}
          lanes={swim.lanes}
          onAdd={(status, lane) => onAdd(swim.newTaskIn(status, lane))}
          onDrop={swim.drop}
          onToggleColumn={collapse.toggle}
          projectNames={swim.projectNames}
        />
      </div>
    )
  }

  return (
    <div
      className={cn('flex flex-1 gap-2 overflow-x-auto px-4 pt-1 pb-3', grabbing && 'cursor-grabbing')}
      onMouseDown={onMouseDown}
      ref={scrollRef}
    >
      {columns.map(col => (
        <Column
          {...cards}
          collapsed={collapse.isCollapsed(col)}
          column={col}
          columns={names}
          key={col.name}
          onAdd={status => onAdd({ status })}
          onDropTask={cards.onMove}
          onToggle={() => collapse.toggle(col)}
        />
      ))}
    </div>
  )
}
