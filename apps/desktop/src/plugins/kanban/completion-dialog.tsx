/** Completion-summary dialog.
 *
 *  The board/drawer let you move a card to `done`, but the backend's
 *  `complete_task` refuses a completion that leaves the card with no result or
 *  summary evidence (`EmptyCompletionError`). A bare status drag therefore
 *  failed with "has no result or summary evidence" and the UI gave no way to
 *  supply it. This collects a short summary and forwards it to the PATCH/bulk
 *  body as `result`/`summary` (parity with `hermes kanban complete --result`).
 */
import { Button, Dialog, DialogContent, DialogFooter, DialogHeader, DialogTitle, Textarea } from '@hermes/plugin-sdk'
import { type ReactNode, useState } from 'react'

export function CompletionSummaryDialog({
  count,
  onCancel,
  onConfirm
}: {
  count: number
  onCancel: () => void
  onConfirm: (summary: string) => void
}) {
  const [text, setText] = useState('')
  const trimmed = text.trim()

  const confirm = () => {
    if (trimmed) {
      onConfirm(trimmed)
    }
  }

  return (
    <Dialog onOpenChange={open => !open && onCancel()} open>
      <DialogContent className="w-[min(32rem,94vw)] max-w-none">
        <DialogHeader>
          <DialogTitle>Completion summary</DialogTitle>
        </DialogHeader>
        <div className="flex flex-col gap-3">
          <p className="text-xs leading-relaxed text-(--ui-text-tertiary)">
            {count > 1
              ? `Stored as the result for ${count} tasks. Required to mark them done.`
              : 'Stored as the task result. Required to mark the task done.'}
          </p>
          <Textarea
            autoFocus
            className="min-h-24 text-[0.8125rem]"
            onChange={event => setText(event.target.value)}
            onKeyDown={event => {
              if ((event.metaKey || event.ctrlKey) && event.key === 'Enter') {
                event.preventDefault()
                confirm()
              }
            }}
            placeholder="Describe what was done…"
            value={text}
          />
        </div>
        <DialogFooter>
          <Button onClick={onCancel} variant="ghost">
            Cancel
          </Button>
          <Button disabled={!trimmed} onClick={confirm}>
            Complete
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

/** A move target that needs completion-summary evidence: any `done` except an
 *  approval out of `review` (which `complete_task` exempts). */
export const needsCompletionSummary = (status: string, fromStatus: string): boolean =>
  status === 'done' && fromStatus !== 'review'

/** Gate a status dispatch behind the completion-summary dialog. Keeps the
 *  branch + dialog state out of the consuming component (board/drawer), so a
 *  `done` move collects a summary and forwards it as `result`/`summary` while a
 *  plain move dispatches straight through. Returns the dialog node to render
 *  and `requestMove(status, fromStatus)` to call where a plain `mutate` was. */
export function useCompletionSummary(
  dispatch: (vars: { result?: string; status: string; summary?: string }) => void
): { dialog: ReactNode; requestMove: (status: string, fromStatus: string) => void } {
  const [pendingStatus, setPendingStatus] = useState<null | string>(null)

  const requestMove = (status: string, fromStatus: string) => {
    if (needsCompletionSummary(status, fromStatus)) {
      setPendingStatus(status)

      return
    }

    dispatch({ status })
  }

  const dialog =
    pendingStatus === null ? null : (
      <CompletionSummaryDialog
        count={1}
        onCancel={() => setPendingStatus(null)}
        onConfirm={summary => {
          dispatch({ result: summary, status: pendingStatus, summary })
          setPendingStatus(null)
        }}
      />
    )

  return { dialog, requestMove }
}
