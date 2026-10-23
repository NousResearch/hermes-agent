import { Button, host, Tip, useValue } from '@hermes/plugin-sdk'
import { useEffect, useRef, useState } from 'react'

import { $boardSlug, $pinnedBoards, kanbanConnectionScope, movePinnedBoard } from './api'
import type { BoardMeta } from './types'
import { useKanban } from './ui'

interface PinnedBoardTabsProps {
  boards: BoardMeta[]
  currentSlug: string
  scope: string
  serverCurrent: string
}

interface BoardDrag {
  scope: string
  slug: string
}

export function PinnedBoardTabs({ boards, currentSlug, scope, serverCurrent }: PinnedBoardTabsProps) {
  const k = useKanban()
  const pinned = useValue($pinnedBoards)
  const liveScope = useValue(host.state.connectionId) ?? 'local'
  const drag = useRef<BoardDrag | null>(null)
  const [dropTarget, setDropTarget] = useState<string | null>(null)

  const visible = pinned.flatMap(slug => {
    const board = boards.find(meta => meta.slug === slug)

    return board ? [board] : []
  })

  const canDrop = () => drag.current?.scope === scope && scope === kanbanConnectionScope()

  const finishDrag = () => {
    drag.current = null
    setDropTarget(null)
  }

  // A native drag belongs to the gateway where it began, including A→B→A.
  // eslint-disable-next-line no-restricted-syntax -- native gesture token, not an atom mirror
  useEffect(() => {
    drag.current = null
    setDropTarget(null)
  }, [liveScope])

  if (!visible.length || scope !== liveScope) {
    return null
  }

  return (
    <div aria-label={k.pinnedBoards} className="flex min-w-0 items-center gap-1 overflow-x-auto" role="group">
      {visible.map((board, index) => (
        <Tip key={board.slug} label={k.pinnedBoardsHint}>
          <Button
            aria-pressed={board.slug === currentSlug}
            className="max-w-36 shrink-0 data-[drop-target=true]:bg-(--ui-control-active-background)"
            data-drop-target={dropTarget === board.slug}
            draggable
            onClick={() => {
              if (scope === kanbanConnectionScope()) {
                $boardSlug.set(board.slug === serverCurrent ? '' : board.slug)
              }
            }}
            onDragEnd={finishDrag}
            onDragOver={event => {
              if (canDrop()) {
                event.preventDefault()
                event.dataTransfer.dropEffect = 'move'
                setDropTarget(board.slug)
              }
            }}
            onDragStart={event => {
              drag.current = { scope, slug: board.slug }
              event.dataTransfer.effectAllowed = 'move'
              event.dataTransfer.setData('application/x-hermes-kanban-board', board.slug)
            }}
            onDrop={event => {
              if (canDrop() && drag.current) {
                event.preventDefault()
                movePinnedBoard(drag.current.slug, board.slug)
              }

              finishDrag()
            }}
            onKeyDown={event => {
              if (!event.altKey || event.ctrlKey || event.metaKey || !['ArrowLeft', 'ArrowRight'].includes(event.key)) {
                return
              }

              event.preventDefault()
              event.stopPropagation()
              const target = visible[index + (event.key === 'ArrowLeft' ? -1 : 1)]

              if (target && scope === kanbanConnectionScope()) {
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
