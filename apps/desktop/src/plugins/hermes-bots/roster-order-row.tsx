/**
 * Visual drop wrapper for reorderable roster rows.
 *
 * Handles dragover detection and renders an insertion line indicator
 * when another bot in the same section and pinned band is dragged above or below this row.
 */

import { cn, useValue } from '@hermes/plugin-sdk'
import type { DragEvent, ReactNode } from 'react'
import { useEffect, useState } from 'react'

import { $draggingBot, BOT_DRAG_MIME, GROUP_DRAG_PREFIX } from './user-sections'

interface ReorderableRosterRowProps {
  children: ReactNode
  draggingPinned: boolean | null
  draggingScope: null | string
  itemKey: string
  onReorder: (sourceKey: string, targetKey: string, position: 'before' | 'after') => void
  pinned: boolean
  scopeKey: string
}

export function ReorderableRosterRow({
  children,
  draggingPinned,
  draggingScope,
  itemKey,
  onReorder,
  pinned,
  scopeKey
}: ReorderableRosterRowProps) {
  const draggingBot = useValue($draggingBot)
  const [dropPosition, setDropPosition] = useState<null | 'before' | 'after'>(null)

  useEffect(() => {
    if (!draggingBot) {
      setDropPosition(null)
    }
  }, [draggingBot])

  const accepts = (event: DragEvent) => {
    return (
      Boolean(draggingBot) &&
      !draggingBot!.startsWith(GROUP_DRAG_PREFIX) &&
      draggingBot !== itemKey &&
      draggingScope === scopeKey &&
      draggingPinned === pinned &&
      event.dataTransfer.types.includes(BOT_DRAG_MIME)
    )
  }

  return (
    <div
      className={cn(
        'relative',
        dropPosition === 'before' &&
          'before:absolute before:-top-0.5 before:left-2 before:right-2 before:z-10 before:h-0.5 before:rounded-full before:bg-(--ui-accent)',
        dropPosition === 'after' &&
          'after:absolute after:-bottom-0.5 after:left-2 after:right-2 after:z-10 after:h-0.5 after:rounded-full after:bg-(--ui-accent)'
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
        const nextPos = event.clientY < rect.top + rect.height / 2 ? 'before' : 'after'

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

        if (!sourceKey || sourceKey.startsWith(GROUP_DRAG_PREFIX) || sourceKey === itemKey) {
          return
        }

        event.preventDefault()
        event.stopPropagation()
        // Reordering can move or remount the source node before dragend fires;
        // clear the in-flight key immediately so the row never stays faded.
        $draggingBot.set(null)

        const rect = event.currentTarget.getBoundingClientRect()
        const position = wasPosition || (event.clientY < rect.top + rect.height / 2 ? 'before' : 'after')

        onReorder(sourceKey, itemKey, position)
      }}
    >
      {children}
    </div>
  )
}
