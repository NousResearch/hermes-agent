import { useStore } from '@nanostores/react'
import { createPortal } from 'react-dom'

import { Codicon } from '@/components/ui/codicon'
import { cn } from '@/lib/utils'
import { $gatewaySwitching } from '@/store/gateway-switch'
import { selectInboxBadge, selectInboxNeedsCount } from '@/store/inbox'

import { InboxPanel } from './inbox-panel'
import { useInbox } from './use-inbox'

const CHIP_CLASS =
  'inline-flex h-full items-center gap-1 rounded-none px-1.5 text-[0.6875rem] text-(--ui-text-tertiary) transition-colors hover:bg-(--chrome-action-hover) hover:text-foreground disabled:cursor-default disabled:opacity-45'

function chipStateText(badge: 'amber' | 'none' | 'red', count: number, errors: number): string {
  if (badge === 'red') {
    return errors > 0 ? `partial — ${errors} read ${errors === 1 ? 'error' : 'errors'}` : 'unsupported / could not read'
  }

  if (badge === 'amber') {return `${count} ${count === 1 ? 'request' : 'requests'} need you`}

  return 'all quiet'
}

/**
 * Bottom status-bar Inbox chip. Single owner of `useInbox()` — drives the
 * poll timer and passes `inbox` down to the panel so no duplicate timers exist.
 * Always shows the count + a descriptive tooltip (never color-only); amber =
 * something needs the operator, red = a read/unsupported error state (never
 * all-clear), muted = nothing pending. Toggling opens/closes the floating panel;
 * neither action resolves anything.
 */
export function InboxStatusbarChip() {
  const { inbox, open, setOpen } = useInbox()
  const switching = useStore($gatewaySwitching)
  const count = selectInboxNeedsCount(inbox)
  const badge = selectInboxBadge(inbox)
  const coverage = inbox.snapshot?.coverage
  const errors = coverage?.errors.length ?? 0
  const scope = coverage ? `${coverage.profile} · ${coverage.connection_scope}` : 'connecting'

  const stateText = !inbox.snapshot && inbox.capability === 'unknown' && !inbox.error
    ? 'connecting'
    : coverage?.partial && !errors ? 'incomplete coverage' : chipStateText(badge, count, errors)

  const label = count > 0 ? `Inbox — ${count} need attention` : 'Agent Inbox'

  return (
    <>
      <button
        aria-label={label}
        aria-pressed={open}
        className={cn(
          CHIP_CLASS,
          open && 'bg-accent/55 text-foreground',
          badge === 'amber' && 'text-amber-600 hover:text-amber-500',
          badge === 'red' && 'text-destructive hover:text-destructive'
        )}
        disabled={switching}
        onClick={() => setOpen(!open)}
        title={`Agent Inbox — ${stateText}. ${scope}. Open the panel to act on requests.`}
        type="button"
      >
        <Codicon name="inbox" size="0.75rem" />
        {count > 0 ? <span className="tabular-nums">{count}</span> : null}
      </button>
      {open ? createPortal(<InboxPanel inbox={inbox} onClose={() => setOpen(false)} />, document.body) : null}
    </>
  )
}
