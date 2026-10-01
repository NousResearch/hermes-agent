/**
 * The chrome a full-page surface is built from — the Kanban board, the
 * Workflows canvas, and whatever page comes next.
 *
 * The point of it being here rather than copied into each page is that two
 * plugins have to read as two pages of ONE app. A page that picks its own
 * background or its own header padding is the tell that it was bolted on, and
 * it's the kind of drift nobody notices while writing the second page and
 * everybody notices when both are open.
 */

import type { ComponentProps, ReactNode } from 'react'

import { cn } from '@/lib/utils'

/**
 * The page root. Owns the app's surface colour and the column the header and
 * body sit in, and it's the positioning context a `SidePanel` pins to — which
 * is why it's `relative` and clips: a panel that slides in from the right
 * shouldn't be able to widen the page on its way.
 */
export function PageShell({ className, ...props }: ComponentProps<'div'>) {
  return (
    <div
      className={cn('relative flex h-full flex-col overflow-hidden bg-(--ui-surface-background)', className)}
      data-slot="page-shell"
      {...props}
    />
  )
}

/**
 * The title row a page opens with.
 *
 * The row height is STATED, not inherited from whichever control happens to be
 * tallest — a page with a search field in its header was 5px taller than one
 * with only icon buttons, and you saw those 5px as a jump every time you moved
 * between the two. `min-h` rather than `h` so a narrow pane can still wrap the
 * row onto a second line; the padding is under the floor, so on one line every
 * page's header is the same height whatever it holds.
 *
 * It's `shrink-0`, so a page whose body is a flex child keeps its header at
 * that height no matter how tall the content gets. Put filters and search
 * inline after the title; put anything that ACTS in `PageHeaderActions`, which
 * pushes itself right.
 */
export function PageHeader({ className, ...props }: ComponentProps<'header'>) {
  return (
    <header
      className={cn('flex min-h-11 shrink-0 flex-wrap items-center gap-2 px-4 py-1.5', className)}
      data-slot="page-header"
      {...props}
    />
  )
}

/** The page's name. One per page — it's the `h1`. */
export function PageHeaderTitle({ className, ...props }: ComponentProps<'h1'>) {
  return (
    <h1 className={cn('text-sm font-semibold text-foreground', className)} data-slot="page-header-title" {...props} />
  )
}

/** How many of the thing the page is about, in a quiet pill beside the title. */
export function PageHeaderCount({ children }: { children: ReactNode }) {
  return (
    <span
      className="rounded-full bg-(--ui-bg-quaternary) px-1.5 py-px text-[0.625rem] tabular-nums text-(--ui-text-tertiary)"
      data-slot="page-header-count"
    >
      {children}
    </span>
  )
}

/** The right-hand cluster. Pushes itself over, so it can sit anywhere in the
 *  header's child order without the caller remembering `ml-auto`. */
export function PageHeaderActions({ className, ...props }: ComponentProps<'div'>) {
  return <div className={cn('ml-auto flex items-center gap-1', className)} data-slot="page-header-actions" {...props} />
}
