import type { ReactNode } from 'react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from '@/components/ui/dropdown-menu'
import { RowButton } from '@/components/ui/row-button'
import { SearchField } from '@/components/ui/search-field'
import { translateNow } from '@/i18n'
import { cn } from '@/lib/utils'

import { OverlayView } from './overlay-view'

// Overlay "panel" primitive — the centered, capped card + framed chrome lifted
// straight from the trace / agents overlay so every non-settings overlay (cron,
// profiles, …) speaks the same visual language: tight type scale, muted
// opacities, NO container borders (rows separate via the row-hover/active bg
// vars + gaps, exactly like the trace waterfall labels).
//
// Compose it as:
//   <Panel onClose>
//     <PanelHeader title subtitle actions={…} />
//     <PanelBody>                 // master/detail row
//       <PanelList>…</PanelList>
//       <PanelDetail>…</PanelDetail>
//     </PanelBody>
//   </Panel>
//
// Single-column views drop their content straight after the header.

interface PanelProps {
  children: ReactNode
  // Root layout override (the card already fills the equidistant inset).
  className?: string
  closeLabel?: string
  contentClassName?: string
  onClose: () => void
}

export function Panel({
  children,
  className,
  closeLabel = translateNow('common.close'),
  contentClassName,
  onClose
}: PanelProps) {
  return (
    <OverlayView
      closeLabel={closeLabel}
      // Top pad aligns the header title's center with the floating close button
      // (which sits at 0.1875rem + titlebar/2, -translate-y-1/2). The X is
      // absolute so it costs no layout space — the header rides up next to it.
      contentClassName={cn(
        'flex h-full min-h-0 flex-col px-4 pb-4 pt-[calc(var(--titlebar-height)/2-0.4375rem)] sm:px-5',
        contentClassName
      )}
      onClose={onClose}
      rootClassName={cn('flex h-full w-full flex-col', className)}
    >
      {children}
    </OverlayView>
  )
}

interface PanelHeaderProps {
  // Right-aligned controls (search, "+ New", segmented control, …).
  actions?: ReactNode
  subtitle?: ReactNode
  title: ReactNode
}

export function PanelHeader({ actions, subtitle, title }: PanelHeaderProps) {
  return (
    <header className="mb-3 flex shrink-0 items-start justify-between gap-3">
      <div className="min-w-0">
        <h2 className="text-sm font-semibold text-foreground">{title}</h2>
        {subtitle ? <p className="truncate text-xs text-muted-foreground/80">{subtitle}</p> : null}
      </div>
      {actions ? <div className="flex shrink-0 items-center gap-1.5">{actions}</div> : null}
    </header>
  )
}

export function PanelBody({ children, className }: { children: ReactNode; className?: string }) {
  return <div className={cn('flex min-h-0 flex-1 gap-5 overflow-hidden', className)}>{children}</div>
}

interface PanelListProps {
  children: ReactNode
  className?: string
  // Pass an onSearchChange to bake a full-bleed filter field in above the items
  // (pinned; the rows scroll under it). Controlled via searchValue.
  onSearchChange?: (value: string) => void
  searchLabel?: string
  searchPlaceholder?: string
  /** Data-derived rotating placeholder nudges (see SearchField.hints). */
  searchHints?: string[]
  searchValue?: string
}

// Left master list. Dense + borderless, like the trace waterfall's label tree:
// single-line rows that touch, separated from the detail only by the body gap.
// An optional search field pins to the top, full-bleed, above the scroll.
export function PanelList({
  children,
  className,
  onSearchChange,
  searchLabel,
  searchPlaceholder,
  searchHints,
  searchValue
}: PanelListProps) {
  return (
    <div className={cn('flex w-52 shrink-0 flex-col', className)}>
      {onSearchChange ? (
        <SearchField
          aria-label={searchLabel ?? searchPlaceholder ?? ''}
          containerClassName="mb-1 w-full shrink-0"
          hints={searchHints}
          onChange={onSearchChange}
          placeholder={searchPlaceholder ?? ''}
          value={searchValue ?? ''}
        />
      ) : null}
      <div className="flex min-h-0 flex-1 flex-col overflow-y-auto overscroll-contain">{children}</div>
    </div>
  )
}

interface PanelListRowProps {
  active: boolean
  // Leading status dot color class (e.g. 'bg-emerald-500'); omit for none.
  dotClassName?: string
  // Leading codicon glyph name (used when there's no lead/dot).
  icon?: string
  // Custom leading element (colored swatch, avatar, …). Wins over dot/icon.
  lead?: ReactNode
  // Trailing per-row kebab menu (pass a <PanelRowMenu/>). Reveals on hover/focus.
  menu?: ReactNode
  // Short always-visible trailing meta (a tag/time, like the trace label's duration).
  meta?: ReactNode
  onSelect: () => void
  rowKey?: string
  title: ReactNode
}

// A row is a container (not a <button>) so it can host both the select target
// and a kebab menu without nesting interactive elements. Hover/active bg lives
// on the wrapper so the whole row highlights as one.
export function PanelListRow({
  active,
  dotClassName,
  icon,
  lead,
  menu,
  meta,
  onSelect,
  rowKey,
  title
}: PanelListRowProps) {
  return (
    <div
      className={cn(
        'group/row row-hover relative flex h-7 w-full items-center rounded-md text-[0.78rem] hover:text-foreground',
        active ? 'bg-(--ui-row-active-background) text-foreground' : 'text-(--ui-text-secondary)'
      )}
      data-panel-row={rowKey}
    >
      <RowButton
        className="flex h-full min-w-0 flex-1 items-center gap-2 rounded-md pl-2 pr-1 text-left"
        onClick={onSelect}
      >
        {lead ??
          (dotClassName ? (
            <span aria-hidden="true" className={cn('size-1.5 shrink-0 rounded-full', dotClassName)} />
          ) : icon ? (
            <Codicon className="shrink-0 text-muted-foreground/55" name={icon} size="0.85rem" />
          ) : null)}
        <span className="min-w-0 flex-1 truncate font-medium text-foreground/85">{title}</span>
      </RowButton>
      {meta ? <span className="shrink-0 pr-2 text-[0.62rem] tabular-nums text-muted-foreground/45">{meta}</span> : null}
      {menu ? <div className="shrink-0 pr-1">{menu}</div> : null}
    </div>
  )
}

export interface PanelMenuItem {
  disabled?: boolean
  icon?: string
  label: string
  onSelect: () => void
  tone?: 'danger' | 'default'
}

// Per-row "⋮" actions menu — mirrors the sidebar session row's settled pattern
// (size-5 ghost trigger + kebab-vertical codicon + w-40 content). Hidden until
// the row is hovered/focused (or the menu is open). Returns null with no items
// (e.g. the default profile, which can't be renamed/deleted).
export function PanelRowMenu({ items, label = 'Actions' }: { items: PanelMenuItem[]; label?: string }) {
  if (items.length === 0) {
    return null
  }

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button
          aria-label={label}
          className="size-5 rounded-[4px] bg-transparent text-(--ui-text-tertiary) opacity-0 transition-colors duration-100 hover:bg-(--ui-control-active-background) hover:text-foreground focus-visible:opacity-100 focus-visible:ring-0 group-hover/row:opacity-100 data-[state=open]:bg-(--ui-control-active-background) data-[state=open]:text-foreground data-[state=open]:opacity-100 [&_svg]:size-3.5!"
          size="icon"
          title={label}
          variant="ghost"
        >
          <Codicon name="kebab-vertical" size="0.875rem" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-40" sideOffset={6}>
        {items.map(item => (
          <DropdownMenuItem
            disabled={item.disabled}
            key={item.label}
            onSelect={item.onSelect}
            variant={item.tone === 'danger' ? 'destructive' : undefined}
          >
            {item.icon ? <Codicon name={item.icon} size="0.875rem" /> : null}
            <span>{item.label}</span>
          </DropdownMenuItem>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

// Scrolling detail region. Fills the column (no right rail here, unlike the
// trace inspector), so the content stretches the full available width.
export function PanelDetail({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <div className={cn('min-h-0 flex-1 overflow-y-auto overscroll-contain', className)}>
      <div className="space-y-4 pb-6 pl-1 pr-2">{children}</div>
    </div>
  )
}

interface PanelEmptyProps {
  action?: ReactNode
  description?: ReactNode
  // Codicon glyph name (e.g. 'hubot', 'warning', 'loading~spin').
  icon?: string
  title?: ReactNode
}

export function PanelEmpty({ action, description, icon = 'inbox', title }: PanelEmptyProps) {
  return (
    <div className="grid flex-1 place-items-center px-6 py-10 text-center">
      <div className="flex flex-col items-center gap-2">
        <Codicon className="text-muted-foreground/50" name={icon} size="1.25rem" />
        {title ? <p className="text-sm font-medium text-foreground/90">{title}</p> : null}
        {description ? (
          <p className="max-w-sm text-xs leading-relaxed text-muted-foreground/70">{description}</p>
        ) : null}
        {action ? <div className="mt-2">{action}</div> : null}
      </div>
    </div>
  )
}

export function PanelSectionLabel({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <div className={cn('text-[0.6rem] font-medium uppercase tracking-wider text-muted-foreground/50', className)}>
      {children}
    </div>
  )
}

// Inspector-style key/value grid (mirrors the trace span inspector's <dl>).
export interface PanelMetaRow {
  label: ReactNode
  value: ReactNode
}

export function PanelMeta({ className, rows }: { className?: string; rows: PanelMetaRow[] }) {
  return (
    <dl className={cn('grid grid-cols-[5rem_1fr] gap-x-2 gap-y-1 text-[0.7rem]', className)}>
      {rows.map((row, i) => (
        <div className="contents" key={typeof row.label === 'string' ? row.label : i}>
          <dt className="truncate text-muted-foreground/55">{row.label}</dt>
          <dd className="min-w-0 break-words text-foreground/85">{row.value}</dd>
        </div>
      ))}
    </dl>
  )
}

// Monospace content block (job prompt, etc.) — mirrors the inspector's
// input/output <pre> blocks: subtle bg, no border.
export function PanelBlock({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <pre
      className={cn(
        'max-h-48 overflow-auto whitespace-pre-wrap break-words rounded bg-foreground/5 p-2.5 text-[0.68rem] leading-relaxed text-foreground/80',
        className
      )}
    >
      {children}
    </pre>
  )
}

export type PanelPillTone = 'bad' | 'good' | 'muted' | 'warn'

const PILL_TONE: Record<PanelPillTone, string> = {
  bad: 'bg-destructive/10 text-destructive',
  good: 'bg-primary/10 text-primary',
  muted: 'bg-foreground/10 text-muted-foreground',
  warn: 'bg-amber-500/10 text-amber-600 dark:text-amber-300'
}

export function PanelPill({ children, tone = 'muted' }: { children: ReactNode; tone?: PanelPillTone }) {
  return (
    <span
      className={cn(
        'inline-flex items-center rounded-full px-1.5 py-0.5 text-[0.62rem] font-medium capitalize',
        PILL_TONE[tone]
      )}
    >
      {children}
    </span>
  )
}

// Self-describing centered "+" that sits as the LAST item in a PanelList. The
// label rides aria/title only — no visible text.
export function PanelAddButton({
  icon = 'add',
  label,
  onClick
}: {
  icon?: string
  label: string
  onClick: () => void
}) {
  return (
    <Button
      aria-label={label}
      className="h-7 w-full shrink-0 justify-center text-muted-foreground/70 hover:bg-(--ui-row-hover-background) hover:text-foreground"
      onClick={onClick}
      size="sm"
      title={label}
      variant="ghost"
    >
      <Codicon name={icon} size="0.875rem" />
    </Button>
  )
}

// Visible ghost action for a detail header (cron pause/resume/trigger, …).
export function PanelAction({
  children,
  disabled,
  icon,
  onClick
}: {
  children: ReactNode
  disabled?: boolean
  icon: string
  onClick: () => void
}) {
  return (
    <Button
      className="gap-1.5 text-muted-foreground hover:bg-(--ui-row-hover-background) hover:text-foreground"
      disabled={disabled}
      onClick={onClick}
      size="sm"
      variant="ghost"
    >
      <Codicon name={icon} size="0.875rem" />
      {children}
    </Button>
  )
}
