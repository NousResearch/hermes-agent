import { useStore } from '@nanostores/react'
import { Fragment, useEffect, useRef } from 'react'

import { $sidebarReorderHint, registerReorderZone } from './reorder-zones'

/**
 * A pointer-drag reorderable list for a flat sidebar section (Pinned).
 *
 * Unlike the dnd-kit `ReorderableList` (which owns its own DndContext and a
 * grab handle), this list has NO handle and NO second drag system: the rows are
 * already session-drag sources (session-drag.ts), and that shared pointer drag
 * now resolves a reorder when the drop lands inside a registered zone. This
 * component is only the registration + the insertion-line UI — dropping a row
 * within the bar reorders, dropping it on the chat links, one gesture routed by
 * where it's released.
 *
 * Items are passed as `{ id, node }` so the list can tag each row for the
 * resolver's geometry snapshot (`data-reorder-row`) and interleave the
 * insertion line at the live hint without reaching into row markup. The zone
 * container is `display: contents` so the row wrappers remain the direct flex
 * children of the section body (unchanged gap/scroll); each wrapper is a real
 * box so its rect reflects the row for slot math.
 */
export function ReorderZoneList({
  items,
  onReorder
}: {
  items: { id: string; node: React.ReactNode }[]
  onReorder: (ids: string[]) => void
}) {
  const containerRef = useRef<HTMLDivElement | null>(null)
  const hint = useStore($sidebarReorderHint)

  // Keep the latest ids/onReorder readable by the (stable) registration without
  // re-registering on every order change — the drag reads them live at engage.
  const idsRef = useRef<string[]>([])
  idsRef.current = items.map(item => item.id)
  const onReorderRef = useRef(onReorder)
  onReorderRef.current = onReorder

  useEffect(() => {
    const el = containerRef.current

    if (!el) {
      return
    }

    return registerReorderZone({
      el,
      getIds: () => idsRef.current,
      onReorder: ids => onReorderRef.current(ids)
    })
  }, [])

  // Only paint the insertion line for a drag that started in THIS list (its id
  // is one of ours) — a chat/tile drag never shows a reorder caret here.
  const showHint = hint !== null && idsRef.current.includes(hint.draggedId)
  const lineBefore = showHint ? hint.before : undefined

  const line = <div aria-hidden className="mx-1 my-px h-0.5 rounded-full bg-primary/70" data-reorder-line />

  return (
    <div className="contents" ref={containerRef}>
      {items.map(item => (
        <Fragment key={item.id}>
          {showHint && lineBefore === item.id ? line : null}
          <div data-reorder-row={item.id}>{item.node}</div>
        </Fragment>
      ))}
      {showHint && lineBefore === null ? line : null}
    </div>
  )
}
