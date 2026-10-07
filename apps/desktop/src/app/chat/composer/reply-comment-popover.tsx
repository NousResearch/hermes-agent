import { useEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'

import { composerPanelCard } from '@/components/chat/composer-dock'
import { Button } from '@/components/ui/button'
import { Kbd } from '@/components/ui/kbd'
import { useI18n } from '@/i18n'
import { cn } from '@/lib/utils'

export interface ReplyCommentPopoverPosition {
  x: number
  y: number
}

const POPOVER_WIDTH = 300
const POPOVER_MARGIN = 12

function clampPosition(position: ReplyCommentPopoverPosition): ReplyCommentPopoverPosition {
  const width = globalThis.innerWidth || 1024
  const height = globalThis.innerHeight || 768

  return {
    x: Math.min(Math.max(POPOVER_MARGIN, position.x), Math.max(POPOVER_MARGIN, width - POPOVER_WIDTH - POPOVER_MARGIN)),
    y: Math.min(Math.max(POPOVER_MARGIN, position.y), Math.max(POPOVER_MARGIN, height - 190))
  }
}

/**
 * Note editor for a pinned reply comment. Enter pins/saves, Shift+Enter is a
 * newline, Esc cancels. Rendered in a portal so a transcript selection can
 * open it without disturbing the message layout.
 */
export function ReplyCommentPopover({
  confirmLabel,
  initialNote = '',
  onClose,
  onConfirm,
  position,
  quote
}: {
  confirmLabel: string
  initialNote?: string
  onClose: () => void
  onConfirm: (note: string) => void
  position: ReplyCommentPopoverPosition
  quote: string
}) {
  const { t } = useI18n()
  const [note, setNote] = useState(initialNote)
  const noteRef = useRef<HTMLTextAreaElement>(null)
  const placed = clampPosition(position)

  useEffect(() => {
    noteRef.current?.focus()
  }, [])

  return createPortal(
    <div
      className={cn(composerPanelCard, 'fixed z-[70] w-[300px] p-2 shadow-xl')}
      data-slot="reply-comment-popover"
      style={{ left: placed.x, top: placed.y }}
    >
      <blockquote className="mb-1.5 line-clamp-3 border-l-2 border-(--ui-accent) pl-1.5 text-[0.72rem] text-muted-foreground">
        {quote}
      </blockquote>
      <textarea
        className="max-h-28 min-h-11 w-full resize-y rounded-md border border-(--border) bg-transparent p-1.5 text-[0.8rem] outline-none placeholder:text-muted-foreground/60 focus:border-(--ui-accent)"
        onChange={event => setNote(event.target.value)}
        onKeyDown={event => {
          if (event.key === 'Escape') {
            event.stopPropagation()
            onClose()
          } else if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault()
            onConfirm(note)
          }
        }}
        placeholder={t.composer.replyComments.notePlaceholder}
        ref={noteRef}
        rows={2}
        value={note}
      />
      <div className="mt-1.5 flex items-center justify-between">
        <span className="flex items-center gap-1 text-[0.65rem] text-muted-foreground/70">
          <Kbd size="sm">Enter</Kbd>
        </span>
        <div className="flex gap-1">
          <Button onClick={onClose} size="sm" type="button" variant="ghost">
            {t.common.cancel}
          </Button>
          <Button onClick={() => onConfirm(note)} size="sm" type="button" variant="secondary">
            {confirmLabel}
          </Button>
        </div>
      </div>
    </div>,
    document.body
  )
}
