import type { ReactNode } from 'react'

import { SCAFFOLD_META_CLASS, ScaffoldRow } from '@/components/chat/scaffold-row'
import { WIDGET_SHELL_CLASS } from '@/components/chat/widget-shell'
import { Button } from '@/components/ui/button'
import { ConnectorLogo, type ConnectorLogoSubject } from '@/components/ui/connector-logo'
import { Check, CircleIcon, Loader2 } from '@/lib/icons'
import { cn } from '@/lib/utils'

/**
 * Presentation leaf: callers own connector semantics and localized copy.
 *
 * The card offers; the transcript explains. One shell, one heading, one row per app. A row carries a
 * mark, a name, at most one cue and at most one verb; it never says why. The verb lives in a fixed
 * lane so every row's control is the same box on the same edge.
 */
export type ConnectorRowMark = 'connected' | 'idle' | 'waiting'

export interface ConnectorRowAction {
  busy?: boolean
  disabled?: boolean
  label: string
  onClick: () => void
}

export interface ConnectorRowProps {
  /** Absent once the row is resolved: there is nothing left to offer. */
  action?: ConnectorRowAction
  connector: ConnectorLogoSubject
  /** The one quiet line after the name; the waiting row's "in your browser" cue. */
  cue?: string
  mark: ConnectorRowMark
  /** What a screen reader gets for the mark. */
  markLabel: string
}

const SHELL_CLASS = `${WIDGET_SHELL_CLASS} text-[length:var(--conversation-text-font-size)] text-(--ui-text-primary)`

const MARKS = {
  connected: { Icon: Check, className: 'text-emerald-600 dark:text-emerald-400' },
  idle: { Icon: CircleIcon, className: 'text-(--ui-text-quaternary)' },
  // The spin is the only motion in the card; a reduced-motion reader keeps the mark, without it.
  waiting: { Icon: Loader2, className: 'animate-spin text-primary motion-reduce:animate-none' }
} satisfies Record<ConnectorRowMark, { Icon: typeof Check; className: string }>

export function ConnectorCard({ children, title }: { children: ReactNode; title: string }) {
  return (
    <div className={cn(SHELL_CLASS, 'my-1.5 grid gap-0.5')} data-slot="connector-card">
      <p className="pb-1.5 font-medium leading-(--conversation-line-height)">{title}</p>
      {children}
    </div>
  )
}

export function ConnectorRow({ action, connector, cue, mark, markLabel }: ConnectorRowProps) {
  const { Icon, className } = MARKS[mark]
  // The mark and the cue are one live region, so a row that flips is announced instead of only seen.
  // The cue often repeats the mark's own word; then it is said once.
  const announcement = cue && cue !== markLabel ? `${markLabel}. ${cue}` : markLabel

  return (
    <div className="grid gap-1" data-connector-row={connector.name} data-slot="connector-row" tabIndex={-1}>
      <div className="flex h-8 items-center gap-2.5">
        <span aria-live="polite" className="grid size-4 shrink-0 place-items-center" role="status">
          <Icon aria-hidden className={cn('size-3.5', className)} />
          <span className="sr-only">{announcement}</span>
        </span>
        <ConnectorLogo className="size-6 rounded-md text-[0.6875rem]" connector={connector} />
        <span className="truncate leading-(--conversation-line-height)">{connector.title || connector.name}</span>
        <span aria-hidden className="min-w-0 flex-1 truncate text-[0.6875rem] text-(--ui-text-tertiary)">
          {cue}
        </span>
        <span className="flex w-22 shrink-0 justify-end">
          {action ? (
            <span className="inline-flex h-6 w-full items-stretch overflow-hidden rounded-md border border-primary/25 bg-primary/10 text-primary">
              <Button
                className="h-full w-full rounded-none px-2 text-xs font-medium text-primary hover:bg-primary/15 hover:text-primary"
                disabled={action.disabled}
                loading={action.busy}
                onClick={action.onClick}
                size="xs"
                variant="ghost"
              >
                {action.label}
              </Button>
            </span>
          ) : null}
        </span>
      </div>

    </div>
  )
}

export function ConnectorSummary({
  connector,
  meta,
  tone
}: {
  connector: ConnectorLogoSubject
  meta?: string
  tone?: 'ok'
}) {
  // Apply opacity to rows, not a shared container: it would create a stacking context for every sibling.
  return (
    <div data-conversation-scaffold="" data-slot="connector-card">
      <ScaffoldRow>
        <ConnectorLogo className="size-4 rounded-[0.25rem]" connector={connector} />
        <span className="truncate text-[length:var(--conversation-tool-font-size)] text-(--ui-text-primary)">
          {connector.title || connector.name}
        </span>
        {meta ? (
          <span className={cn(SCAFFOLD_META_CLASS, tone === 'ok' && 'text-emerald-600/85 dark:text-emerald-400/85')}>
            {meta}
          </span>
        ) : null}
      </ScaffoldRow>
    </div>
  )
}
