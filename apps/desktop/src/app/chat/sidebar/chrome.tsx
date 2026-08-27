import { useStore } from '@nanostores/react'
import type * as React from 'react'

import { Codicon } from '@/components/ui/codicon'
import { DisclosureCaret } from '@/components/ui/disclosure-caret'
import { RowButton } from '@/components/ui/row-button'
import { Tip } from '@/components/ui/tooltip'
import { compactNumber } from '@/lib/format'
import { cn } from '@/lib/utils'
import { $sidebarRowMeta } from '@/store/layout'

import {
  SIDEBAR_ROW_INSET,
  SIDEBAR_ROW_LABEL,
  SIDEBAR_ROW_LEAD,
  SIDEBAR_ROW_MIN_H,
  SIDEBAR_ROW_PAD_TRAIL
} from './row-geometry'

// Shared, content-agnostic sidebar chrome — used by both the flat session
// sections and the project/workspace tree, so it lives outside either to keep
// imports one-directional (no index <-> projects cycle).

/** The muted slot beside a section label (loading glyph, status hint). */
export function SidebarSectionMeta({ children }: { children: React.ReactNode }) {
  return <span className="shrink-0 text-[0.6875rem] font-medium text-(--ui-text-quaternary)">{children}</span>
}

// Row geometry lives in `row-geometry.ts` — see that file for why each class
// belongs to the box it belongs to. Re-exported here because this module is
// where callers already look for row chrome.
export { SIDEBAR_LEAD_ICON_SIZE, SIDEBAR_ROW_CARD_MIN_H, SIDEBAR_TRUNCATED_LEADING } from './row-geometry'

/** Vertical stack of rows (gap-px, single column). */
export function SidebarRowStack({ className, ...props }: React.ComponentProps<'div'>) {
  return <div className={cn('grid grid-cols-[minmax(0,1fr)] gap-px', className)} {...props} />
}

/** Nested rows (session previews, worktree bodies). */
export function SidebarRowNest({ className, ...props }: React.ComponentProps<'div'>) {
  return <SidebarRowStack className={cn('pb-1 pl-2', className)} {...props} />
}

/**
 * Chronological date-bucket separator ("Yesterday" / "Last week" / "June") for
 * the session list. Caption-shaped — a small label plus a hairline — so it
 * groups sessions by recency without adding a level of indentation. When
 * `toggle` is set the whole caption collapses the sessions beneath it, same
 * gesture as a repo header (the hover caret is the tell).
 */
export function SidebarDateDivider({
  action,
  className,
  label,
  toggle,
  ...props
}: React.ComponentProps<'div'> & {
  action?: React.ReactNode
  label: string
  toggle?: { ariaLabel: string; onToggle: () => void; open: boolean }
}) {
  const caption = (
    <span className="shrink-0 text-[0.64rem] font-semibold uppercase tracking-[0.12em] text-(--ui-text-quaternary)">
      {label}
    </span>
  )
  const rule = <span aria-hidden="true" className="h-px min-w-4 flex-1 bg-(--ui-stroke-tertiary)" />

  return (
    // group/workspace: a divider heads a group the same way a repo header does,
    // so it borrows the header's hover-revealed "+" verbatim.
    <div className={cn('group/workspace flex select-none items-center gap-2 px-2 pb-0.5 pt-2', className)} {...props}>
      {toggle ? (
        <button
          aria-expanded={toggle.open}
          aria-label={toggle.ariaLabel}
          className="flex min-w-0 flex-1 items-center gap-2 bg-transparent text-left"
          onClick={toggle.onToggle}
          type="button"
        >
          {caption}
          <DisclosureCaret
            className="text-(--ui-text-tertiary) opacity-0 transition group-hover/workspace:opacity-100"
            open={toggle.open}
          />
          {rule}
        </button>
      ) : (
        <>
          {caption}
          {rule}
        </>
      )}
      {action}
    </div>
  )
}

/** Outer grid — sole owner of row height and of the trailing inset. The
 *  `actions` slot is marked `data-row-actions` so a row-wide drag gesture can
 *  exclude it with one selector: it holds real controls, never grab surface.
 *  It stretches so that exclusion covers the column, not just its tallest
 *  control. */
export function SidebarRowShell({
  actions,
  actionsClassName,
  children,
  className,
  ...props
}: React.ComponentProps<'div'> & { actions?: React.ReactNode; actionsClassName?: string }) {
  return (
    <div
      className={cn(
        SIDEBAR_ROW_MIN_H,
        SIDEBAR_ROW_PAD_TRAIL,
        'grid grid-cols-[minmax(0,1fr)_auto] items-stretch rounded-md',
        className
      )}
      {...props}
    >
      {children}
      {actions ? (
        <div className={cn('flex shrink-0 items-center self-stretch', actionsClassName)} data-row-actions>
          {actions}
        </div>
      ) : null}
    </div>
  )
}

/** Multi-control left cluster (project rows). */
export function SidebarRowCluster({ className, ...props }: React.ComponentProps<'div'>) {
  return <div className={cn(SIDEBAR_ROW_INSET, className)} {...props} />
}

/** Session row main tap target. */
export function SidebarRowBody({ className, ...props }: React.ComponentProps<'button'>) {
  return <RowButton className={cn(SIDEBAR_ROW_INSET, 'bg-transparent text-left', className)} {...props} />
}

/** Tappable label — underline/truncate live on the inner span, not the button. */
export function SidebarRowLink({
  className,
  labelClassName,
  children,
  ...props
}: React.ComponentProps<'button'> & { labelClassName?: string }) {
  return (
    <RowButton className={cn('min-w-0 shrink bg-transparent p-0 text-left', className)} {...props}>
      <span className={cn(SIDEBAR_ROW_LABEL, labelClassName)}>{children}</span>
    </RowButton>
  )
}

/** Fixed leading column (dot, icon, drag handle). */
export function SidebarRowLead({ className, ...props }: React.ComponentProps<'span'>) {
  return <span className={cn(SIDEBAR_ROW_LEAD, className)} {...props} />
}

/** Standard row label typography. */
export function SidebarRowLabel({ className, ...props }: React.ComponentProps<'span'>) {
  return <span className={cn(SIDEBAR_ROW_LABEL, className)} {...props} />
}

/** What a group's sessions add up to, for the Show options that count something. */
export interface SidebarGroupTotals {
  costUsd: number
  tokens: number
}

/**
 * Header for a group of sessions that hangs its rows underneath — a project, a
 * profile. Row-shaped rather than caption-shaped (that's {@link SidebarDateDivider},
 * which still collapses, just without a lead glyph), so a group header lines up
 * with the session rows it heads. `toggle` omitted keeps the caret's space with
 * nothing to reveal.
 */
export function SidebarGroupRow({
  actions,
  className,
  label,
  lead,
  toggle,
  totals,
  ...props
}: React.ComponentProps<'div'> & {
  actions?: React.ReactNode
  label: React.ReactNode
  lead: React.ReactNode
  toggle?: { ariaLabel: string; onToggle: () => void; open: boolean }
  totals?: SidebarGroupTotals
}) {
  const rowMeta = useStore($sidebarRowMeta)

  const facts = [
    totals && rowMeta.includes('tokens') && totals.tokens > 0 ? compactNumber(totals.tokens) : null,
    // Sub-cent spend rounds to "$0.00", which reads as a bug rather than as a
    // cheap group — below a cent the header says nothing at all.
    totals && rowMeta.includes('cost') && totals.costUsd >= 0.01 ? `$${totals.costUsd.toFixed(2)}` : null
  ].filter(Boolean) as string[]

  return (
    <SidebarRowShell
      actions={
        // The controls overlay the figures rather than sitting beside them: in
        // flow they hold their width open at all times, which reads as a gap
        // torn between the total and the row's edge. Same trade the session row
        // makes with its kebab and age — you read the number or you act on the
        // group, never both at once.
        facts.length ? (
          <div className="relative flex items-center">
            <span className="min-w-9 whitespace-nowrap text-right text-[0.625rem] leading-none text-(--ui-text-tertiary) transition-opacity group-hover/workspace:opacity-0">
              {facts.join(' · ')}
            </span>
            {actions ? <div className="absolute right-0 flex items-center">{actions}</div> : null}
          </div>
        ) : (
          actions
        )
      }
      className={cn('group/workspace', className)}
      {...props}
    >
      <SidebarRowCluster className="min-w-0 flex-1">
        {lead}
        {label}
        {toggle ? (
          <Tip label={toggle.ariaLabel}>
            <button
              aria-label={toggle.ariaLabel}
              className="flex flex-1 items-center self-stretch bg-transparent p-0"
              data-row-actions
              onClick={toggle.onToggle}
              type="button"
            >
              <DisclosureCaret
                className="shrink-0 text-(--ui-text-tertiary) opacity-0 transition group-hover/workspace:opacity-100"
                open={toggle.open}
              />
            </button>
          </Tip>
        ) : (
          <span className="flex-1" />
        )}
      </SidebarRowCluster>
    </SidebarRowShell>
  )
}

/** Dot ↔ grabber swap for dnd-kit reorder rows. */
export function SidebarRowGrab({
  ariaLabel,
  children,
  className,
  dragging = false,
  dragHandleProps,
  leadClassName
}: {
  ariaLabel: string
  children: React.ReactNode
  className?: string
  dragging?: boolean
  dragHandleProps?: React.HTMLAttributes<HTMLElement>
  leadClassName?: string
}) {
  return (
    <SidebarRowLead
      {...dragHandleProps}
      aria-label={ariaLabel}
      className={cn(
        'group/handle relative cursor-grab touch-none overflow-hidden active:cursor-grabbing',
        leadClassName,
        className
      )}
      data-reorder-handle
      onClick={event => event.stopPropagation()}
    >
      <span className="grid size-full place-items-center transition-opacity group-hover/handle:opacity-0 group-focus-within/handle:opacity-0">
        {children}
      </span>
      <Codicon
        className={cn(
          'absolute text-(--ui-text-quaternary) opacity-0 transition-opacity group-hover/handle:opacity-80 group-focus-within/handle:opacity-80 hover:text-(--ui-text-secondary)',
          dragging && 'text-(--ui-text-secondary) opacity-100'
        )}
        name="grabber"
        size="0.75rem"
      />
    </SidebarRowLead>
  )
}

/** Icon/dot slot inside SidebarRowLead — caps visual size so rows align. */
export function SidebarRowLeadGlyph({
  children,
  className,
  style
}: {
  children: React.ReactNode
  className?: string
  style?: React.CSSProperties
}) {
  return (
    <span
      className={cn('grid size-full place-items-center text-(--ui-text-tertiary) [&_.codicon]:leading-none', className)}
      style={style}
    >
      {children}
    </span>
  )
}
