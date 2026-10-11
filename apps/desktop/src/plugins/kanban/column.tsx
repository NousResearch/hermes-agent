/** One column of the flat board: header, cards (optionally sub-grouped by
 *  profile in Running), the dashed "+", and its collapsed rail. */

import { cn, useValue } from '@hermes/plugin-sdk'
import { useMemo } from 'react'

import { $lanesByProfile } from './api'
import { Card } from './card'
import { ColumnAddButton, ColumnDot, ColumnHeaderBar, useDropTarget } from './column-parts'
import type { KanbanTask } from './types'
import { Avatar, columnLabel, isLockedTarget, useKanban } from './ui'

const UNASSIGNED_LANE = 'unassigned'

export function Column({
  collapsed,
  column,
  columns,
  onAdd,
  onDelete,
  onDropTask,
  onMove,
  onOpen,
  onToggle,
  onToggleSelect,
  selected
}: {
  collapsed: boolean
  column: { name: string; tasks: KanbanTask[] }
  columns: string[]
  onAdd: (status: string) => void
  onDelete: (id: string) => void
  onDropTask: (id: string, status: string) => void
  onMove: (id: string, status: string) => void
  onOpen: (id: string) => void
  onToggle: () => void
  onToggleSelect: (id: string) => void
  selected: ReadonlySet<string>
}) {
  const k = useKanban()
  const label = columnLabel(k, column.name)
  const locked = isLockedTarget(column.name)
  const byProfile = useValue($lanesByProfile)

  // The dashboard's "lanes by profile": sub-group Running by assignee so a
  // fleet's in-flight work reads per-worker. Null = flat (off, or trivial).
  const lanes = useMemo(() => {
    if (!byProfile || column.name !== 'running' || column.tasks.length === 0) {
      return null
    }

    const groups = new Map<string, KanbanTask[]>()

    for (const task of column.tasks) {
      const key = task.assignee || UNASSIGNED_LANE
      groups.set(key, [...(groups.get(key) ?? []), task])
    }

    return [...groups.entries()].sort(([a], [b]) => a.localeCompare(b))
  }, [byProfile, column])

  // Locked lanes refuse the drop — the lane is honest about itself.
  const { handlers: dragHandlers, wash } = useDropTarget(
    () => !locked,
    id => onDropTask(id, column.name)
  )

  // Collapsed = a thin vertical rail: dot, sideways label, count. Still a live
  // drop target (drop straight onto the rail); click expands. The dot sits in
  // the same h-5 header row as an expanded lane's, so dots align across the
  // board regardless of collapse state.
  if (collapsed) {
    return (
      <button
        {...dragHandlers}
        aria-label={k.expand(label)}
        className={cn(
          'flex h-full w-8 shrink-0 flex-col items-center gap-1.5 rounded-lg p-2 transition-colors hover:bg-(--ui-bg-quinary)',
          wash
        )}
        onClick={onToggle}
        type="button"
      >
        <span className="grid h-5 shrink-0 place-items-center">
          <ColumnDot name={column.name} />
        </span>
        <span className="text-[0.6875rem] font-medium uppercase tracking-wide text-(--ui-text-tertiary) [writing-mode:vertical-rl]">
          {label}
        </span>
        {column.tasks.length > 0 && (
          <span className="text-[0.625rem] tabular-nums text-(--ui-text-quaternary)">{column.tasks.length}</span>
        )}
      </button>
    )
  }

  return (
    <div
      {...dragHandlers}
      className={cn('group/col flex h-full w-64 shrink-0 flex-col rounded-lg p-2 transition-colors', wash)}
    >
      <ColumnHeaderBar className="mb-1.5 px-1" count={column.tasks.length} name={column.name} onCollapse={onToggle} />
      <div className="relative flex min-h-0 flex-1 flex-col gap-2 overflow-y-auto">
        {lanes
          ? lanes.map(([assignee, tasks]) => (
              <div className="flex flex-col gap-2" key={assignee}>
                <div className="flex items-center gap-1.5 px-1 pt-1 text-[0.625rem] text-(--ui-text-quaternary)">
                  {assignee !== UNASSIGNED_LANE && <Avatar name={assignee} size="0.875rem" />}
                  {assignee}
                  <span className="tabular-nums">{tasks.length}</span>
                </div>
                {tasks.map(task => (
                  <Card
                    columns={columns}
                    key={task.id}
                    onDelete={onDelete}
                    onMove={onMove}
                    onOpen={onOpen}
                    onToggleSelect={onToggleSelect}
                    selected={selected.has(task.id)}
                    task={task}
                  />
                ))}
              </div>
            ))
          : column.tasks.map(task => (
              <Card
                columns={columns}
                key={task.id}
                onDelete={onDelete}
                onMove={onMove}
                onOpen={onOpen}
                onToggleSelect={onToggleSelect}
                selected={selected.has(task.id)}
                task={task}
              />
            ))}
        <ColumnAddButton name={column.name} onAdd={() => onAdd(column.name)} />
        {column.tasks.length === 0 && (
          <div className="pointer-events-none absolute inset-0 grid place-items-center text-[0.6875rem] text-(--ui-text-quaternary)">
            {k.empty}
          </div>
        )}
      </div>
    </div>
  )
}
