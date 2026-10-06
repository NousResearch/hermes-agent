import type { DragEvent } from 'react'

import { $draggingBot, BOT_DRAG_MIME } from './user-sections'

/** Keep native roster drags out of the file tree's global react-dnd backend. */
export function startRosterDrag(event: Pick<DragEvent, 'stopPropagation' | 'dataTransfer'>, key: string): void {
  event.stopPropagation()
  event.dataTransfer.setData(BOT_DRAG_MIME, key)
  event.dataTransfer.effectAllowed = 'move'
  $draggingBot.set(key)
}
