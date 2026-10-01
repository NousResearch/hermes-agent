/** The board split into horizontal swimlanes: one sticky column-header row,
 *  then one row of cells per lane. Every cell is a drop target for its
 *  (lane, column) pair; what a cross-lane drop changes is decided by
 *  `dropPatch` in ./swimlanes, and the page refuses what it can't apply. */

import { cn, Codicon, Tip, useValue } from '@hermes/plugin-sdk'
import { type DragEvent as ReactDragEvent, type ReactNode, useRef, useState } from 'react'

import { $collapsedSwimlanes } from './api'
import { Card } from './card'
import { type LaneDimension, type Swimlane } from './swimlanes'
import { columnMeta, type KanbanColumn } from './types'
import { Avatar, columnHelp, columnLabel, isLockedTarget, type KanbanText, useKanban } from './ui'

const LANE_TITLE: Record<
  LaneDimension,
  (key: string, k: KanbanText, projectNames: ReadonlyMap<string, string>) => string
> = {
  // A project id this profile's projects.db doesn't know (archived, or
  // created by another profile) still gets its own lane, under its id.
  project: (key, k, names) => (key ? (names.get(key) ?? key) : k.noProjectLane),
  assignee: (key, k) => key || k.unassigned,
  tenant: (key, k) => key || k.noTenant,
  priority: (key, k) => k.priorityLane(Number(key))
}

/** Display name of a lane — also the sort key for named dimensions. */
export const laneTitle = (
  by: LaneDimension,
  key: string,
  k: KanbanText,
  projectNames: ReadonlyMap<string, string>
): string => LANE_TITLE[by](key, k, projectNames)

const COLUMN_WIDTH = 'w-64'
const RAIL_WIDTH = 'w-8'

interface CardActions {
  onDelete: (id: string) => void
  onMove: (id: string, status: string) => void
  onOpen: (id: string) => void
  onToggleSelect: (id: string) => void
  selected: ReadonlySet<string>
}

export interface SwimlaneGridProps extends CardActions {
  by: LaneDimension
  /** The (filtered) board columns — header order and per-column totals. */
  columns: KanbanColumn[]
  lanes: Swimlane[]
  collapsedColumn: (column: KanbanColumn) => boolean
  onToggleColumn: (column: KanbanColumn) => void
  projectNames: ReadonlyMap<string, string>
  onAdd: (status: string, lane: string) => void
  canDrop: (id: string, lane: string, status: string) => boolean
  onDrop: (id: string, lane: string, status: string) => void
}

export function SwimlaneGrid(props: SwimlaneGridProps) {
  const { by, collapsedColumn, columns, lanes, projectNames } = props
  const k = useKanban()
  const collapsedLanes = useValue($collapsedSwimlanes)
  // The dragged card's id, captured as the drag starts (dataTransfer is
  // unreadable during dragover) so cells can refuse a drop up front.
  const dragId = useRef<null | string>(null)
  const names = columns.map(col => col.name)
  // Collapse is a property of the whole column, judged on its board-wide total.
  const collapsedNames = new Set(columns.filter(collapsedColumn).map(col => col.name))

  const toggleLane = (key: string) => {
    const id = `${by}:${key}`
    const next = { ...collapsedLanes }

    if (next[id]) {
      delete next[id]
    } else {
      next[id] = true
    }

    $collapsedSwimlanes.set(next)
  }

  return (
    <div
      className="inline-flex min-w-full flex-col gap-3"
      onDragEnd={() => (dragId.current = null)}
      onDragStart={event => (dragId.current = event.dataTransfer.getData('text/plain') || null)}
    >
      <div className="sticky top-0 z-[2] flex gap-2 bg-(--ui-surface-background) pb-1">
        {columns.map(col => (
          <ColumnHeader
            collapsed={collapsedNames.has(col.name)}
            column={col}
            key={col.name}
            onToggle={() => props.onToggleColumn(col)}
          />
        ))}
      </div>
      {lanes.map(lane => {
        const collapsed = Boolean(collapsedLanes[`${by}:${lane.key}`])
        const title = laneTitle(by, lane.key, k, projectNames)

        return (
          <section className="flex flex-col gap-1.5" key={lane.key}>
            <LaneHeader
              collapsed={collapsed}
              count={lane.count}
              label={
                by === 'assignee' && lane.key ? (
                  <>
                    <Avatar name={lane.key} size="0.875rem" />
                    {title}
                  </>
                ) : (
                  title
                )
              }
              onToggle={() => toggleLane(lane.key)}
              title={title}
            />
            {!collapsed && (
              <div className="flex gap-2">
                {lane.columns.map(col => (
                  <Cell
                    {...props}
                    collapsed={collapsedNames.has(col.name)}
                    column={col}
                    columnNames={names}
                    dragId={dragId}
                    key={col.name}
                    lane={lane.key}
                  />
                ))}
              </div>
            )}
          </section>
        )
      })}
    </div>
  )
}

function ColumnHeader({
  collapsed,
  column,
  onToggle
}: {
  collapsed: boolean
  column: KanbanColumn
  onToggle: () => void
}) {
  const k = useKanban()
  const meta = columnMeta(column.name)
  const label = columnLabel(k, column.name)
  const dot = <span className="size-1.5 shrink-0 rounded-full" style={{ backgroundColor: meta.tone }} />

  if (collapsed) {
    return (
      <Tip label={label}>
        <button
          aria-label={k.expand(label)}
          className={cn(
            RAIL_WIDTH,
            'flex h-5 shrink-0 items-center justify-center gap-1 rounded hover:bg-(--ui-bg-quinary)'
          )}
          onClick={onToggle}
          type="button"
        >
          {dot}
        </button>
      </Tip>
    )
  }

  return (
    <header className={cn(COLUMN_WIDTH, 'group/col flex h-5 shrink-0 items-center gap-1.5 px-3')}>
      {dot}
      <Tip label={columnHelp(k, column.name)}>
        <span className="cursor-help text-[0.6875rem] font-medium uppercase tracking-wide text-(--ui-text-tertiary)">
          {label}
        </span>
      </Tip>
      <span className="text-[0.625rem] tabular-nums text-(--ui-text-quaternary)">{column.tasks.length}</span>
      <button
        aria-label={k.collapse(label)}
        className="ml-auto grid size-5 place-items-center rounded text-(--ui-text-tertiary) opacity-0 transition-opacity hover:bg-(--chrome-action-hover) hover:text-foreground focus-visible:opacity-100 group-hover/col:opacity-100"
        onClick={onToggle}
        type="button"
      >
        <Codicon name="chevron-left" size="0.75rem" />
      </button>
    </header>
  )
}

function LaneHeader({
  collapsed,
  count,
  label,
  onToggle,
  title
}: {
  collapsed: boolean
  count: number
  label: ReactNode
  onToggle: () => void
  title: string
}) {
  const k = useKanban()

  return (
    // Sticky-left so the lane's name stays readable while the columns scroll
    // sideways under it.
    <button
      aria-expanded={!collapsed}
      aria-label={collapsed ? k.expand(title) : k.collapse(title)}
      className="sticky left-0 flex w-fit max-w-full items-center gap-1.5 rounded px-1 py-0.5 text-[0.75rem] font-medium text-(--ui-text-secondary) hover:bg-(--chrome-action-hover) hover:text-foreground"
      onClick={onToggle}
      type="button"
    >
      <Codicon name={collapsed ? 'chevron-right' : 'chevron-down'} size="0.75rem" />
      <span className="flex min-w-0 items-center gap-1.5 truncate">{label}</span>
      <span className="text-[0.625rem] tabular-nums text-(--ui-text-quaternary)">{count}</span>
    </button>
  )
}

function Cell({
  canDrop,
  collapsed,
  column,
  columnNames,
  dragId,
  lane,
  onAdd,
  onDelete,
  onDrop,
  onMove,
  onOpen,
  onToggleSelect,
  selected
}: SwimlaneGridProps & {
  collapsed: boolean
  column: KanbanColumn
  columnNames: string[]
  dragId: { current: null | string }
  lane: string
}) {
  const k = useKanban()
  const [over, setOver] = useState(false)
  const locked = isLockedTarget(column.name)
  const label = columnLabel(k, column.name)

  const dragHandlers = {
    onDragLeave: () => setOver(false),
    onDragOver: (event: ReactDragEvent<HTMLElement>) => {
      // Not calling preventDefault refuses the drop: the OS shows no-drop and
      // the drop event never fires.
      if (locked || (dragId.current !== null && !canDrop(dragId.current, lane, column.name))) {
        event.dataTransfer.dropEffect = 'none'

        return
      }

      event.preventDefault()
      event.dataTransfer.dropEffect = 'move'
      setOver(true)
    },
    onDrop: (event: ReactDragEvent<HTMLElement>) => {
      event.preventDefault()
      setOver(false)
      const id = event.dataTransfer.getData('text/plain')

      if (id) {
        onDrop(id, lane, column.name)
      }
    }
  }

  const wash = over ? 'bg-(--ui-bg-quinary)' : 'bg-[color-mix(in_srgb,var(--ui-bg-quinary)_50%,transparent)]'

  if (collapsed) {
    return (
      <div
        {...dragHandlers}
        className={cn(RAIL_WIDTH, 'flex shrink-0 justify-center rounded-lg py-2 transition-colors', wash)}
      >
        {column.tasks.length > 0 && (
          <span className="text-[0.625rem] tabular-nums text-(--ui-text-quaternary)">{column.tasks.length}</span>
        )}
      </div>
    )
  }

  return (
    <div
      {...dragHandlers}
      className={cn(
        COLUMN_WIDTH,
        'group/col flex min-h-16 shrink-0 flex-col gap-2 rounded-lg p-2 transition-colors',
        wash
      )}
    >
      {column.tasks.map(task => (
        <Card
          columns={columnNames}
          key={task.id}
          onDelete={onDelete}
          onMove={onMove}
          onOpen={onOpen}
          onToggleSelect={onToggleSelect}
          selected={selected.has(task.id)}
          task={task}
        />
      ))}
      {!locked && (
        <button
          aria-label={k.newTaskIn(label)}
          className="flex shrink-0 items-center justify-center rounded-md border border-dashed border-(--ui-stroke-secondary) py-1.5 text-(--ui-text-tertiary) opacity-0 transition-[opacity,color,border-color] group-hover/col:opacity-100 hover:border-(--ui-text-quaternary) hover:bg-(--chrome-action-hover) hover:text-foreground focus-visible:opacity-100"
          onClick={() => onAdd(column.name, lane)}
          type="button"
        >
          <Codicon name="add" size="0.8rem" />
        </button>
      )}
    </div>
  )
}
