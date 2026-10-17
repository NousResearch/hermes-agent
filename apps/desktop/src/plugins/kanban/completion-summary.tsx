/**
 * Completing a card needs evidence. The backend refuses a move to Done that
 * carries no summary on a card with no stored result (`_gate_empty_completion`;
 * a review approval is exempt). Board drag, the bulk bar and the drawer's
 * status menu therefore ask for one up front instead of bouncing the move.
 */

import {
  Button,
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  isSubmitEnter,
  Textarea
} from '@hermes/plugin-sdk'
import { type ReactNode, useCallback, useRef, useState } from 'react'

import type { KanbanTask } from './types'
import { useKanban } from './ui'

/** True when moving `task` to `target` would be refused for lack of evidence. */
export const needsCompletionSummary = (task: Pick<KanbanTask, 'result' | 'status'>, target: string): boolean =>
  target === 'done' && task.status !== 'review' && !task.result?.trim()

function SummaryDialog({ count, onSettle }: { count: number; onSettle: (summary: null | string) => void }) {
  const k = useKanban()
  const [text, setText] = useState('')
  const summary = text.trim()

  return (
    <Dialog onOpenChange={open => !open && onSettle(null)} open>
      <DialogContent className="w-[min(32rem,94vw)] max-w-none">
        <DialogHeader>
          <DialogTitle>{k.completeTitle(count)}</DialogTitle>
        </DialogHeader>
        <Textarea
          autoFocus
          className="min-h-24"
          onChange={event => setText(event.target.value)}
          onKeyDown={event => {
            if (isSubmitEnter(event) && !event.shiftKey && summary) {
              event.preventDefault()
              onSettle(summary)
            }
          }}
          placeholder={k.completePlaceholder}
          value={text}
        />
        <DialogFooter>
          <Button onClick={() => onSettle(null)} variant="text">
            {k.cancel}
          </Button>
          <Button disabled={!summary} onClick={() => onSettle(summary)}>
            {k.complete}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

/**
 * `ask(count)` opens the dialog and resolves with the typed summary, or `null`
 * when the user cancels. Render `dialog` once beside the surface that calls it.
 */
export function useCompletionSummary(): { ask: (count?: number) => Promise<null | string>; dialog: ReactNode } {
  const [open, setOpen] = useState<null | { count: number; id: number }>(null)
  const pending = useRef<((summary: null | string) => void) | null>(null)
  const seq = useRef(0)

  const settle = useCallback((summary: null | string) => {
    pending.current?.(summary)
    pending.current = null
    setOpen(null)
  }, [])

  const ask = useCallback(
    (count = 1) =>
      new Promise<null | string>(resolve => {
        pending.current?.(null)
        pending.current = resolve
        setOpen({ count, id: (seq.current += 1) })
      }),
    []
  )

  // `key` resets the typed text between asks.
  return { ask, dialog: open && <SummaryDialog count={open.count} key={open.id} onSettle={settle} /> }
}
