/**
 * The card's comment composer.
 */

import { Button, Codicon, isSubmitEnter, Textarea, Tip } from '@hermes/plugin-sdk'
import { useState } from 'react'

import { useKanban } from './ui'

// Mirrors the review pane's commit-message field: one row tall to start
// (button-height), CSS field-sizing grows it with content, button hugs the
// bottom edge as it grows.
//
// On a RUNNING task the worker polls its comment thread and folds new notes
// into the live turn (OUT-OF-BAND steer), so a plain note reaches the agent
// mid-run within a few seconds — no block/unblock dance. `onRequeue` is the
// heavier option: post the note AND reclaim so the task restarts from scratch
// with the note in context (use when the current run has gone off the rails).
export function CommentComposer({
  onRequeue,
  onSubmit,
  pending,
  running
}: {
  onRequeue?: (body: string) => void
  onSubmit: (body: string) => void
  pending: boolean
  running?: boolean
}) {
  const k = useKanban()
  const [body, setBody] = useState('')

  const submit = () => {
    const trimmed = body.trim()

    if (trimmed && !pending) {
      onSubmit(trimmed)
      setBody('')
    }
  }

  const requeue = () => {
    const trimmed = body.trim()

    if (trimmed && !pending && onRequeue) {
      onRequeue(trimmed)
      setBody('')
    }
  }

  const empty = !body.trim() || pending
  const sendLabel = running ? k.send : k.comment

  // Sized like the new-project idea field (default control padding, inset icon
  // action); a labelled text button floating inside the textarea read as part
  // of the input.
  return (
    <div className="flex flex-col gap-1.5">
      <div className="relative">
        <Textarea
          className="field-sizing-content max-h-40 resize-none pr-9 text-[0.8125rem]"
          onChange={event => setBody(event.target.value)}
          onKeyDown={event => {
            if (isSubmitEnter(event) && !event.shiftKey) {
              event.preventDefault()
              submit()
            }
          }}
          placeholder={running ? k.messageWorker : k.addComment}
          value={body}
        />
        <Tip label={sendLabel}>
          <Button
            aria-label={sendLabel}
            className="absolute top-1 right-1 text-muted-foreground/80 hover:text-foreground"
            disabled={empty}
            onClick={submit}
            size="icon-xs"
            type="button"
            variant="ghost"
          >
            <Codicon name="arrow-up" size="0.85rem" />
          </Button>
        </Tip>
      </div>
      {running && onRequeue && (
        <div className="flex items-center justify-between gap-2">
          <span className="text-[0.625rem] leading-tight text-(--ui-text-quaternary)">{k.deliveredLive}</span>
          <Button className="shrink-0" disabled={empty} onClick={requeue} size="xs" variant="outline">
            <Codicon name="debug-restart" size="0.7rem" />
            {k.requeueWithNote}
          </Button>
        </div>
      )}
    </div>
  )
}
