import type { ReactNode } from 'react'

import { DisclosureCaret } from '@/components/ui/disclosure-caret'
import { cn } from '@/lib/utils'

// Shared header row for any collapsible block (thinking, tool group, single
// tool). Each parent supplies its own outer wrapper (with the data-slot CSS
// uses to escape the message padding) and its own expanded body.
//
// Affordance:
//   - No leading chevron; a caret appears to the RIGHT of the text on hover
//     (and stays visible when the row is open).
//   - The hover background is a tight content-shaped pill — sized to the
//     title text, NOT the full row — and reaches just past the chevron with
//     `-mx-1.5 px-1.5` so it reads as a soft hit-target rather than a slab
//     stretching to the message edge.
//   - `trailing` stays in flow (e.g. a duration timer), so the title always
//     reserves space for it instead of painting underneath it. Interactive
//     controls go in `action`, which lays out *in flow* at the far right so it
//     never sits on top of the caret's hit-target.
export function DisclosureRow({
  action,
  children,
  onToggle,
  open,
  trailing
}: {
  action?: ReactNode
  children: ReactNode
  onToggle?: () => void
  open: boolean
  trailing?: ReactNode
}) {
  return (
    <div className="group/disclosure-row relative flex w-full max-w-full min-w-0 text-(--ui-text-tertiary)">
      <button
        aria-expanded={onToggle ? open : undefined}
        className={cn(
          // max-w-fit so the click target hugs the title text width — no
          // background fill, just the cursor + the affordance caret.
          'flex min-w-0 max-w-fit items-start gap-1.5 text-left transition-colors',
          onToggle ? 'hover:text-foreground focus-visible:text-foreground focus-visible:outline-none' : 'cursor-default'
        )}
        disabled={!onToggle}
        onClick={onToggle}
        type="button"
      >
        <span className="flex min-w-0 flex-col gap-0.5">{children}</span>
        {onToggle && (
          // Wrapper height matches the title row's actual line-height so the
          // caret centres with the title, not the whole subtitle stack.
          <span
            className={cn(
              'flex h-(--conversation-line-height) shrink-0 items-center justify-center transition-opacity duration-150',
              open
                ? 'opacity-80'
                : 'opacity-(--disclosure-caret-rest) group-hover/disclosure-row:opacity-80 group-focus-within/disclosure-row:opacity-80'
            )}
          >
            <DisclosureCaret open={open} />
          </span>
        )}
      </button>
      {action && (
        <span className="ml-auto flex h-(--conversation-line-height) shrink-0 items-center self-start pl-1.5">
          {action}
        </span>
      )}
      {trailing && <span className="flex h-(--conversation-line-height) shrink-0 items-center pl-1.5">{trailing}</span>}
    </div>
  )
}
