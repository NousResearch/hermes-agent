/**
 * Visual drop wrapper for reorderable roster rows.
 *
 * Handles dragover detection and renders an insertion line indicator
 * when another bot in the same section and pinned band is dragged above or below this row.
 */

import { cn, useValue } from '@hermes/plugin-sdk'
import type { DragEvent, ReactNode } from 'react'
import { useEffect, useState } from 'react'

import { $draggingBot, $draggingBotPinned, $draggingBotScope, BOT_DRAG_MIME } from './roster-order'

interface ReorderableRosterRowProps {
  children: ReactNode
  itemKey: string
  onReorder: (sourceKey: string, targetKey: string, position: 'before' | 'after') => void
  pinned: boolean
  scopeKey: string
}

export function ReorderableRosterRow({ children, itemKey, onReorder, pinned, scopeKey }: ReorderableRosterRowProps) {
  const draggingBot = useValue($draggingBot)
  const draggingScope = useValue($draggingBotScope)
  const draggingPinned = useValue($draggingBotPinned)
  const [dropPosition, setDropPosition] = useState<null | 'before' | 'after'>(null)

  useEffect(() => {
    if (!draggingBot) {
      setDropPosition(null)
    }
  }, [draggingBot])

  const accepts = (event: DragEvent) => {
    return (
      Boolean(draggingBot) &&
      draggingBot !== itemKey &&
      draggingScope === scopeKey &&
      draggingPinned === pinned &&
      event.dataTransfer.types.includes(BOT_DRAG_MIME)
    )
  }

  return (
    <div
      className={cn(
        'relative transition-all duration-75',
        dropPosition === 'before' &&
          'before:absolute before:-top-0.5 before:left-2 before:right-2 before:h-0.5 before:rounded-full before:bg-(--ui-accent) before:z-10',
        dropPosition === 'after' &&
          'after:absolute after:-bottom-0.5 after:left-2 after:right-2 after:h-0.5 after:rounded-full after:bg-(--ui-accent) after:z-10'
      )}
      data-roster-item-key={itemKey}
      onDragLeave={event => {
        if (!event.currentTarget.contains(event.relatedTarget as Node | null)) {
          setDropPosition(null)
        }
      }}
      onDragOver={event => {
        if (!accepts(event)) {
          return
        }

        event.preventDefault()
        event.dataTransfer.dropEffect = 'move'

        const rect = event.currentTarget.getBoundingClientRect()
        const midY = rect.top + rect.height / 2
        const nextPos = event.clientY < midY ? 'before' : 'after'

        if (dropPosition !== nextPos) {
          setDropPosition(nextPos)
        }
      }}
      onDrop={event => {
        const wasPosition = dropPosition
        setDropPosition(null)

        if (!accepts(event)) {
          return
        }

        const sourceKey = event.dataTransfer.getData(BOT_DRAG_MIME) || draggingBot

        if (!sourceKey || sourceKey === itemKey) {
          return
        }

        event.preventDefault()
        event.stopPropagation()

        const rect = event.currentTarget.getBoundingClientRect()
        const midY = rect.top + rect.height / 2
        const position = wasPosition || (event.clientY < midY ? 'before' : 'after')

        onReorder(sourceKey, itemKey, position)
      }}
    >
      {children}
    </div>
  )
}
