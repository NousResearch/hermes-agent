import { useStore } from '@nanostores/react'
import { useState } from 'react'

import { requestComposerSubmit } from '@/app/chat/composer/focus'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { Textarea } from '@/components/ui/textarea'
import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import { isSubmitEnter } from '@/lib/ime'
import { formatCombo } from '@/lib/keybinds/combo'
import { cn } from '@/lib/utils'
import { notifyError } from '@/store/notifications'
import {
  $reviewCommentDraft,
  $reviewComments,
  $reviewScopeTarget,
  addReviewComment,
  cancelReviewCommentDraft,
  clearReviewComments,
  formatReviewCommentsPrompt,
  removeReviewComment
} from '@/store/review'

const ICON = '0.85rem'

/** Line comments under the selected file's diff: the draft editor for the
 *  clicked diff row, the running list across files, and the one-click hand-off
 *  that sends every comment to the owning chat's agent (the ship bar's
 *  requestComposerSubmit path). Renders nothing until a line is clicked or a
 *  comment exists, so the pane is unchanged for non-commenting users. */
export function ReviewComments() {
  const { t } = useI18n()
  const c = t.statusStack.coding
  const comments = useStore($reviewComments)
  const draft = useStore($reviewCommentDraft)
  const scopeTarget = useStore($reviewScopeTarget)
  const [body, setBody] = useState('')

  if (!draft && comments.length === 0) {
    return null
  }

  const submitDraft = () => {
    if (!draft || !body.trim()) {
      return
    }

    addReviewComment({ ...draft, body: body.trim() })
    setBody('')
  }

  const sendToAgent = () => {
    if (comments.length === 0) {
      return
    }

    if (!requestComposerSubmit(formatReviewCommentsPrompt(comments), { target: scopeTarget })) {
      notifyError(new Error(c.agentShipUnavailable), c.sendComments)

      return
    }

    clearReviewComments()
  }

  return (
    <div className="flex shrink-0 flex-col gap-1.5 border-t border-(--ui-stroke-secondary) p-2" data-suppress-pane-reveal-side="">
      {draft && (
        <div className="flex flex-col gap-1">
          <div className="truncate font-mono text-[0.64rem] text-(--ui-text-tertiary)" title={draft.snippet}>
            {c.commentOnLine(draft.path, draft.line)}
          </div>
          <div className="relative">
            <Textarea
              autoFocus
              className="field-sizing-content max-h-32 min-h-0 resize-none pr-9"
              onChange={event => setBody(event.target.value)}
              onKeyDown={event => {
                if ((event.metaKey || event.ctrlKey) && isSubmitEnter(event)) {
                  event.preventDefault()
                  submitDraft()
                }

                if (event.key === 'Escape') {
                  cancelReviewCommentDraft()
                  setBody('')
                }
              }}
              placeholder={c.commentPlaceholder(formatCombo('mod+enter'))}
              value={body}
            />
            <span className="absolute inset-y-0 right-1 flex items-start pt-1">
              <Tip label={c.addComment}>
                <Button
                  aria-label={c.addComment}
                  className="size-6"
                  disabled={!body.trim()}
                  onClick={submitDraft}
                  size="icon-xs"
                  variant="ghost"
                >
                  <Codicon name="check" size={ICON} />
                </Button>
              </Tip>
            </span>
          </div>
        </div>
      )}

      {comments.length > 0 && (
        <>
          <div className="flex items-center gap-1">
            <span className="min-w-0 flex-1 text-[0.66rem] font-medium text-(--ui-text-secondary)">
              {c.commentsCount(comments.length)}
            </span>
            <Tip label={c.clearComments}>
              <Button
                aria-label={c.clearComments}
                className="size-6"
                onClick={clearReviewComments}
                size="icon-xs"
                variant="ghost"
              >
                <Codicon name="trash" size={ICON} />
              </Button>
            </Tip>
          </div>
          <div className="flex max-h-44 flex-col gap-1 overflow-auto">
            {comments.map(comment => (
              <div
                className="group flex items-start gap-1 rounded-md bg-(--ui-control-background) px-1.5 py-1"
                key={comment.id}
              >
                <div className="min-w-0 flex-1">
                  <div className="truncate font-mono text-[0.62rem] text-(--ui-text-tertiary)">
                    {comment.path}:{comment.line}
                  </div>
                  <div className={cn('text-[0.68rem] wrap-break-word text-(--ui-text-secondary)')}>{comment.body}</div>
                </div>
                <Tip label={c.removeComment}>
                  <Button
                    aria-label={c.removeComment}
                    className="size-5 shrink-0 opacity-0 group-hover:opacity-100"
                    onClick={() => removeReviewComment(comment.id)}
                    size="icon-xs"
                    variant="ghost"
                  >
                    <Codicon name="close" size="0.7rem" />
                  </Button>
                </Tip>
              </div>
            ))}
          </div>
          <Button
            className="justify-center text-[0.7rem] text-muted-foreground/85 hover:text-foreground"
            onClick={sendToAgent}
            size="sm"
            variant="ghost"
          >
            <span className="truncate underline underline-offset-2">{c.sendComments}</span>
          </Button>
        </>
      )}
    </div>
  )
}
