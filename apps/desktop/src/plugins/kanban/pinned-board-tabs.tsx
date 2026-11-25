import { Button, Tip, useValue } from '@hermes/plugin-sdk'
import { useRef, useState } from 'react'

import { $boardSlug, $pinnedBoards, kanbanConnectionScope, movePinnedBoard } from './api'
import type { BoardMeta } from './types'
import { useKanban } from './ui'

/** One-click tabs for pinned boards, beside the switcher. Drag or Alt+Left/Right
 *  reorders; selection only changes on click. Pins for boards that no longer
 *  exist stay stored but hidden, so a board that comes back keeps its slot. */
export function PinnedBoardTabs({
  boards,
  currentSlug,
  serverCurrent
}: {
  boards: BoardMeta[]
  currentSlug: string
  serverCurrent: string
}) {
  const k = useKanban()
  const pinned = useValue($pinnedBoards)
  // The connection a drag began on: a drop after a connection switch must not
  // reorder the other gateway's pins.
  const drag = useRef<{ scope: string; slug: string } | null>(null)
  const [dropTarget, setDropTarget] = useState<string | null>(null)

  const visible = pinned.flatMap(slug => boards.filter(meta => meta.slug === slug))

  if (!visible.length) {
    return null
  }

  const canDrop = () => drag.current?.scope === kanbanConnectionScope()

  const endDrag = () => {
    drag.current = null
    setDropTarget(null)
  }

  return (
    <div aria-label={k.pinnedBoards} className="flex min-w-0 items-center gap-1 overflow-x-auto" role="group">
      {visible.map((board, index) => (
        <Tip key={board.slug} label={k.pinnedBoardsHint}>
          <Button
            aria-pressed={board.slug === currentSlug}
            className="max-w-36 data-[drop-target=true]:bg-(--ui-control-active-background)"
            data-drop-target={dropTarget === board.slug}
            draggable
            onClick={() => $boardSlug.set(board.slug === serverCurrent ? '' : board.slug)}
            onDragEnd={endDrag}
            onDragOver={event => {
              if (canDrop()) {
                event.preventDefault()
                event.dataTransfer.dropEffect = 'move'
                setDropTarget(board.slug)
              }
            }}
            onDragStart={event => {
              drag.current = { scope: kanbanConnectionScope(), slug: board.slug }
              event.dataTransfer.effectAllowed = 'move'
            }}
            onDrop={event => {
              if (canDrop() && drag.current) {
                event.preventDefault()
                movePinnedBoard(drag.current.slug, board.slug)
              }

              endDrag()
            }}
            onKeyDown={event => {
              const step = event.key === 'ArrowLeft' ? -1 : event.key === 'ArrowRight' ? 1 : 0
              const target = visible[index + step]

              if (event.altKey && step && target) {
                event.preventDefault()
                movePinnedBoard(board.slug, target.slug)
              }
            }}
            size="xs"
            variant={board.slug === currentSlug ? 'secondary' : 'ghost'}
          >
            <span className="truncate">{board.name || board.slug}</span>
          </Button>
        </Tip>
      ))}
    </div>
  )
}
