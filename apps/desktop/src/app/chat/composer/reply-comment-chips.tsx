import { useState } from 'react'
import { useStore } from '@nanostores/react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import { cn } from '@/lib/utils'

import { ReplyCommentPopover, type ReplyCommentPopoverPosition } from './reply-comment-popover'
import type { ReplyComment } from './reply-comments'
import { useComposerScope } from './scope'

const QUOTE_SNIPPET_CHARS = 90

const snippet = (text: string): string =>
  text.length <= QUOTE_SNIPPET_CHARS ? text : `${text.slice(0, QUOTE_SNIPPET_CHARS).trimEnd()}…`

/**
 * Pinned reply comments above the composer — one chip per quote+note. Edit
 * reopens the note popover, × drops the chip. Everything left pinned freezes
 * into the message as Markdown blocks on the next send.
 */
export function ReplyCommentChips() {
  const { t } = useI18n()
  const scope = useComposerScope()
  const copy = t.composer.replyComments
  const comments = useStore(scope.replyComments.$comments)
  const [editing, setEditing] = useState<(ReplyComment & ReplyCommentPopoverPosition) | null>(null)

  if (comments.length === 0 && !editing) {
    return null
  }

  return (
    <>
      {comments.length > 0 && (
        <div className="flex max-w-full flex-shrink-0 flex-wrap gap-1.5 px-1 pt-1" data-slot="reply-comment-chips">
          {comments.map(comment => (
            <span
              className="inline-flex max-w-full items-center gap-1 rounded-md border border-(--ui-accent)/40 bg-(--ui-accent)/10 py-0.5 pl-1.5 pr-0.5 text-[0.72rem]"
              key={comment.id}
              title={comment.note ? `${comment.quote}\n— ${comment.note}` : comment.quote}
            >
              <Codicon className="shrink-0 opacity-70" name="comment" size="0.75rem" />
              <span className="truncate">{comment.note || snippet(comment.quote)}</span>
              <Tip label={copy.editComment}>
                <Button
                  aria-label={copy.editComment}
                  className={cn('h-5 w-5 rounded p-0 text-muted-foreground hover:text-foreground')}
                  onClick={event => {
                    const rect = (event.currentTarget as HTMLElement).getBoundingClientRect()
                    setEditing({ ...comment, x: Math.max(8, rect.left - 140), y: Math.max(8, rect.top - 190) })
                  }}
                  size="icon"
                  type="button"
                  variant="ghost"
                >
                  <Codicon name="edit" size="0.7rem" />
                </Button>
              </Tip>
              <Tip label={copy.removeComment}>
                <Button
                  aria-label={copy.removeComment}
                  className={cn('h-5 w-5 rounded p-0 text-muted-foreground hover:text-foreground')}
                  onClick={() => {
                    if (editing?.id === comment.id) {
                      setEditing(null)
                    }

                    scope.replyComments.remove(comment.id)
                  }}
                  size="icon"
                  type="button"
                  variant="ghost"
                >
                  <Codicon name="close" size="0.7rem" />
                </Button>
              </Tip>
            </span>
          ))}
        </div>
      )}
      {editing && (
        <ReplyCommentPopover
          confirmLabel={copy.save}
          initialNote={editing.note}
          key={editing.id}
          onClose={() => setEditing(null)}
          onConfirm={note => {
            scope.replyComments.update(editing.id, { note: note.replace(/\s+/g, ' ').trim().slice(0, 500) })
            setEditing(null)
          }}
          position={editing}
          quote={editing.quote}
        />
      )}
    </>
  )
}
