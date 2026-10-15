/** Column chrome shared by the flat board's columns and the swimlane grid's
 *  cells: the header row, the drop-target behaviour, and the dashed "+". */

import { cn, Codicon, Tip } from '@hermes/plugin-sdk'
import { type DragEvent as ReactDragEvent, useState } from 'react'

import { columnMeta } from './types'
import { columnHelp, columnLabel, isLockedTarget, useKanban } from './ui'

/** Drop-target wiring for a column or cell. A refused target never calls
 *  preventDefault, so the OS shows no-drop and `drop` never fires. */
export function useDropTarget(accepts: () => boolean, onDropId: (id: string) => void) {
  const [over, setOver] = useState(false)

  const handlers = {
    onDragLeave: () => setOver(false),
    onDragOver: (event: ReactDragEvent<HTMLElement>) => {
      if (!accepts()) {
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
        onDropId(id)
      }
    }
  }

  const wash = over ? 'bg-(--ui-bg-quinary)' : 'bg-[color-mix(in_srgb,var(--ui-bg-quinary)_50%,transparent)]'

  return { handlers, wash }
}

export function ColumnDot({ name }: { name: string }) {
  return <span className="size-1.5 shrink-0 rounded-full" style={{ backgroundColor: columnMeta(name).tone }} />
}

/** An expanded column's header: dot, label (help on hover), count, collapse. */
export function ColumnHeaderBar({
  className,
  count,
  name,
  onCollapse
}: {
  className?: string
  count: number
  name: string
  onCollapse: () => void
}) {
  const k = useKanban()
  const label = columnLabel(k, name)

  return (
    <header className={cn('flex h-5 items-center gap-1.5', className)}>
      <ColumnDot name={name} />
      <Tip label={columnHelp(k, name)}>
        <span className="cursor-help text-[0.6875rem] font-medium uppercase tracking-wide text-(--ui-text-tertiary)">
          {label}
        </span>
      </Tip>
      <span className="text-[0.625rem] tabular-nums text-(--ui-text-quaternary)">{count}</span>
      <button
        aria-label={k.collapse(label)}
        className="ml-auto grid size-5 place-items-center rounded text-(--ui-text-tertiary) opacity-0 transition-opacity hover:bg-(--chrome-action-hover) hover:text-foreground focus-visible:opacity-100 group-hover/col:opacity-100"
        onClick={onCollapse}
        type="button"
      >
        <Codicon name="chevron-left" size="0.75rem" />
      </button>
    </header>
  )
}

/** Jira-style add — dashed, faded in on column hover. Opacity (not display)
 *  so it always holds its slot and never thrashes layout. Locked columns get
 *  none: you can't create into a system state. */
export function ColumnAddButton({ name, onAdd }: { name: string; onAdd: () => void }) {
  const k = useKanban()

  if (isLockedTarget(name)) {
    return null
  }

  return (
    <button
      aria-label={k.newTaskIn(columnLabel(k, name))}
      className="flex shrink-0 items-center justify-center rounded-md border border-dashed border-(--ui-stroke-secondary) py-1.5 text-(--ui-text-tertiary) opacity-0 transition-[opacity,color,border-color] group-hover/col:opacity-100 hover:border-(--ui-text-quaternary) hover:bg-(--chrome-action-hover) hover:text-foreground focus-visible:opacity-100"
      onClick={onAdd}
      type="button"
    >
      <Codicon name="add" size="0.8rem" />
    </button>
  )
}
