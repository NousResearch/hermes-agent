// The one place the canvas asks YOU something.
//
// A human step is the only kind whose output the run can't produce, so it's the
// only one that gets a dialog. Everything else the canvas has to say it says on
// a card or in the feed; interrupting for those would be noise. Interrupting
// for this is the point — the run is stopped until it's answered.
//
// Modal on purpose. A question you can lose behind the window it's about is a
// question that silently holds the run forever, which is the bug this replaces.
// Escape and clicking away defer instead of dismissing: the run panel keeps a
// way back, so deferring is "let me look first", never "never mind".

import { Button, Codicon, Dialog, DialogContent, DialogFooter, DialogHeader, DialogTitle } from '@hermes/plugin-sdk'

import type { Question } from './player'

export function AskDialog({
  title,
  who,
  prompt,
  onFail,
  open,
  onDefer,
  onRespond
}: Question & {
  title: string
  open: boolean
  onDefer: () => void
  onRespond: (decision: 'approved' | 'denied') => void
}) {
  return (
    <Dialog onOpenChange={o => !o && onDefer()} open={open}>
      <DialogContent aria-describedby={undefined} className="max-w-sm" showCloseButton={false}>
        <DialogHeader>
          <span className="flex items-center gap-1.5 text-[0.6875rem] leading-4 text-(--ui-text-tertiary)">
            <Codicon name="bell" size="0.75rem" />
            waiting on {who}
          </span>
          <DialogTitle>{title}</DialogTitle>
        </DialogHeader>
        <p className="text-xs leading-5 text-(--ui-text-secondary)">{prompt}</p>
        <DialogFooter>
          <Button onClick={() => onRespond('denied')} size="sm" variant="secondary">
            {onFail === 'retry' ? 'Send back' : 'Deny'}
          </Button>
          <Button autoFocus onClick={() => onRespond('approved')} size="sm">
            Approve
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
