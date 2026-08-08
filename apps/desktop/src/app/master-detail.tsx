import { useStore } from '@nanostores/react'
import { type ReactNode, type PointerEvent as ReactPointerEvent, useEffect, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from '@/components/ui/dropdown-menu'
import { RowButton } from '@/components/ui/row-button'
import { Switch } from '@/components/ui/switch'
import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import { cn } from '@/lib/utils'
import { $paneHeightOverride, $paneState, setPaneHeightOverride } from '@/store/panes'

// Monospace capability chip (tool name, transport, …). Shared by the Skills
// and MCP tabs so the pill reads identically everywhere.
export function ToolChip({ children, title }: { children: ReactNode; title?: string }) {
  return (
    <span
      className="rounded-[var(--radius-sm)] bg-(--ui-bg-quinary) px-1.5 py-0.5 font-mono text-[0.65rem] text-(--ui-text-tertiary)"
      title={title}
    >
      {children}
    </span>
  )
}

// Master–detail page scaffolding: one calm, bordered work surface with a
// compact rail and a roomy inspector. Shared by Capabilities and Messaging;
// pages bring their own row/detail content (CapRow here is the toggle-row
// flavor; Messaging has its own avatar rows).

// `pane` docks a full-bleed work surface (editor, log viewer, terminal) below
// the whole master–detail grid — the app's bottom-pane pattern, page-local.
// The wide-rail track shared by every Capabilities tab (skills/tools/mcp) so
// the three read as one page. Exported for pages that build their own grid
// (the MCP tab's cursor-driven layout) but must stay in step.
export const MASTER_DETAIL_WIDE_COLS = 'sm:grid-cols-[minmax(0,0.75fr)_minmax(0,1fr)]'

// `split="wide"` gives list-heavy pages a rail that shares the page with a
// sparse detail (skills/tools/mcp); the default 14rem rail suits pages whose
// detail carries the weight (messaging).
export function MasterDetail({
  children,
  pane,
  split = 'rail'
}: {
  children: ReactNode
  pane?: ReactNode
  split?: 'rail' | 'wide'
}) {
  return (
    <div className="flex h-full min-h-0 flex-col px-[clamp(1.25rem,4vw,4rem)] py-5">
      <div
        className={cn(
          'flex min-h-0 flex-1 flex-col overflow-hidden rounded-[var(--radius-xl)] border border-(--ui-stroke-secondary) bg-[color-mix(in_srgb,var(--ui-bg-elevated)_72%,transparent)] shadow-[0_1px_2px_rgb(0_0_0/0.04)]'
        )}
      >
        <div
          className={cn(
            'grid min-h-0 flex-1 grid-cols-1',
            split === 'wide' ? MASTER_DETAIL_WIDE_COLS : 'sm:grid-cols-[14rem_minmax(0,1fr)]'
          )}
        >
          {children}
        </div>
        {pane}
      </div>
    </div>
  )
}

export function ListColumn({ children, header }: { children: ReactNode; header?: ReactNode }) {
  return (
    <aside className="flex min-h-0 flex-col border-b border-(--ui-stroke-quaternary) bg-[color-mix(in_srgb,var(--ui-bg-tertiary)_36%,transparent)] p-2.5 sm:border-b-0 sm:border-r sm:p-3">
      {header}
      <div className="min-h-0 flex-1 overflow-y-auto overscroll-contain [scrollbar-gutter:stable]">{children}</div>
    </aside>
  )
}

// `footer` pins one quiet caption below the scroll (e.g. "changes apply to
// new sessions") so per-item detail components never repeat it themselves.
// `actionBar` pins a real control row (save/toggle) below the scroll instead.
export function DetailColumn({
  actionBar,
  children,
  footer
}: {
  actionBar?: ReactNode
  children: ReactNode
  footer?: ReactNode
}) {
  return (
    <main className="flex min-h-0 flex-col overflow-hidden bg-[color-mix(in_srgb,var(--ui-bg-elevated)_48%,transparent)]">
      <div className="min-h-0 flex-1 overflow-y-auto overscroll-contain [scrollbar-gutter:stable]">
        <div className="mx-auto w-full max-w-3xl space-y-7 px-5 py-6 sm:px-7">{children}</div>
      </div>
      {footer && (
        <div className="mx-auto w-full max-w-2xl shrink-0 px-5 pb-3 pt-2 text-right text-[0.65rem] text-muted-foreground/50">
          {footer}
        </div>
      )}
      {actionBar && (
        <footer className="shrink-0 border-t border-(--ui-stroke-quaternary) bg-[color-mix(in_srgb,var(--ui-bg-elevated)_86%,transparent)] px-5 py-3">
          <div className="mx-auto flex max-w-2xl flex-wrap items-center gap-2">{actionBar}</div>
        </footer>
      )}
    </main>
  )
}

// Full-bleed docked bottom pane: title strip + actions + close, drag-resizable
// on its top edge like every other pane (height persisted through the same
// pane-state store the terminal uses). No min height — drag (or the chevron)
// collapses it down to just the header. Content swaps freely: JSON editor
// today, stdio/log viewers tomorrow.
const DETAIL_PANE_DEFAULT_BODY_PX = 288
const DETAIL_PANE_MAX_VH = 0.7
const DETAIL_PANE_COLLAPSED_PX = 4

// Ghost icon-button on the kebab-trigger scale (pane headers, list-strip menu,
// per-server MCP actions, JSON editor format button). MUST stay a class string
// (not a CSS @utility): the leading `size-5` is what tailwind-merge uses to
// strip <Button size="icon">'s larger built-in size — a custom utility class
// isn't size-merge-aware, so Button's icon size would leak and blow it up.
// Compose extra state (data-[state=open], hover:text-destructive) with cn().
export const ICON_BUTTON =
  'size-5 cursor-pointer rounded-[var(--radius-sm)] text-muted-foreground/70 transition-[background-color,color,transform] duration-150 hover:bg-(--ui-control-active-background) hover:text-foreground active:scale-95'

export function DetailPane({
  actions,
  children,
  defaultCollapsed = false,
  defaultHeight = DETAIL_PANE_DEFAULT_BODY_PX,
  id,
  onClose,
  title
}: {
  actions?: ReactNode
  children: ReactNode
  /** Start collapsed to the header the first time this pane is ever shown.
   *  Only seeds when the id has no saved state — a later expand/collapse
   *  persists and wins, so it's "collapsed by default", not "always collapsed". */
  defaultCollapsed?: boolean
  /** Default body height in px (before any user resize). */
  defaultHeight?: number
  /** Pane-store key — height overrides persist under it. */
  id: string
  /** Omit for permanent panes (collapsible to the header, never removed). */
  onClose?: () => void
  title: ReactNode
}) {
  const { t } = useI18n()
  const override = useStore($paneHeightOverride(id))

  useEffect(() => {
    if (defaultCollapsed && $paneState(id).get() === undefined) {
      setPaneHeightOverride(id, 0)
    }
  }, [defaultCollapsed, id])

  const height = override ?? defaultHeight
  const collapsed = height <= DETAIL_PANE_COLLAPSED_PX
  // Sash drag mirrors the shell's y-axis pane resize: pointer capture on the
  // top edge, clamped to [0, 70vh]; double-click resets to the default.
  const [dragging, setDragging] = useState(false)

  const startDrag = (event: ReactPointerEvent<HTMLDivElement>) => {
    event.preventDefault()
    const startY = event.clientY
    const startHeight = height
    const max = Math.round(window.innerHeight * DETAIL_PANE_MAX_VH)
    setDragging(true)

    const onMove = (move: globalThis.PointerEvent) => {
      setPaneHeightOverride(id, Math.min(max, Math.max(0, Math.round(startHeight + (startY - move.clientY)))))
    }

    const onUp = () => {
      window.removeEventListener('pointermove', onMove)
      setDragging(false)
    }

    window.addEventListener('pointermove', onMove)
    window.addEventListener('pointerup', onUp, { once: true })
  }

  return (
    <section className="relative flex shrink-0 flex-col border-t border-(--ui-stroke-tertiary) bg-(--ui-chat-surface-background)">
      <div
        className="group/sash absolute inset-x-0 top-0 z-10 h-1 -translate-y-1/2 cursor-row-resize"
        onDoubleClick={() => setPaneHeightOverride(id, undefined)}
        onPointerDown={startDrag}
      >
        <div
          className={cn(
            'absolute inset-x-0 top-1/2 h-px -translate-y-1/2 transition-colors',
            dragging ? 'bg-(--ui-stroke-secondary)' : 'group-hover/sash:bg-(--ui-stroke-secondary)'
          )}
        />
      </div>
      <header className="flex h-9 shrink-0 items-center gap-2 px-3">
        <span className="min-w-0 truncate text-xs font-medium text-foreground">{title}</span>
        <div className="ml-auto flex shrink-0 items-center gap-1.5">
          {actions}
          <Tip label={collapsed ? t.common.expand : t.common.collapse}>
            <Button
              aria-expanded={!collapsed}
              aria-label={collapsed ? t.common.expand : t.common.collapse}
              className={ICON_BUTTON}
              onClick={() => setPaneHeightOverride(id, collapsed ? undefined : 0)}
              size="icon"
              variant="ghost"
            >
              <Codicon name={collapsed ? 'chevron-up' : 'chevron-down'} size="0.8125rem" />
            </Button>
          </Tip>
          {onClose && (
            <Button aria-label={t.common.close} className={ICON_BUTTON} onClick={onClose} size="icon" variant="ghost">
              <Codicon name="close" size="0.8125rem" />
            </Button>
          )}
        </div>
      </header>
      <div className="min-h-0 overflow-hidden" style={{ height: collapsed ? 0 : height }}>
        {children}
      </div>
    </section>
  )
}

// One-line control strip pinned above the list: sort/primary action on the
// left, overflow kebab on the right.
export function ListStrip({
  label,
  left,
  meta,
  right
}: {
  label?: ReactNode
  left?: ReactNode
  meta?: ReactNode
  right?: ReactNode
}) {
  return (
    <div className="mb-2 flex h-7 shrink-0 items-center justify-between gap-2 px-1">
      <div className="flex min-w-0 items-center gap-1.5">
        {label && (
          <span className="truncate text-[0.72rem] font-semibold tracking-[0.01em] text-foreground">{label}</span>
        )}
        {meta !== undefined && (
          <span className="rounded-[var(--radius-sm)] bg-(--ui-bg-quinary) px-1.5 py-0.5 text-[0.62rem] tabular-nums text-(--ui-text-tertiary)">
            {meta}
          </span>
        )}
        {left}
      </div>
      <div className="flex shrink-0 items-center gap-1.5">{right}</div>
    </div>
  )
}

export interface ListStripMenuItem {
  disabled?: boolean
  label: string
  onSelect: () => void
}

export interface ListStripMenuToggle {
  checked: boolean
  disabled?: boolean
  label: string
  onToggle: (checked: boolean) => void
}

// Overflow kebab for list-wide actions. `toggle` renders as the first row —
// one label + switch line covering enable-all/disable-all (checked = every
// visible item on; mixed reads as off so one flip always means "all on").
export function ListStripMenu({
  items = [],
  label,
  toggle
}: {
  items?: ListStripMenuItem[]
  label: string
  toggle?: ListStripMenuToggle
}) {
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button
          aria-label={label}
          className={cn(
            ICON_BUTTON,
            'data-[state=open]:bg-(--ui-control-active-background) data-[state=open]:text-foreground'
          )}
          size="icon"
          variant="ghost"
        >
          <Codicon name="kebab-vertical" size="0.8125rem" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-44" sideOffset={6}>
        {toggle && (
          <DropdownMenuItem
            disabled={toggle.disabled}
            onSelect={event => {
              // Keep the menu open so the switch is seen flipping.
              event.preventDefault()
              toggle.onToggle(!toggle.checked)
            }}
          >
            <span className="min-w-0 flex-1 truncate">{toggle.label}</span>
            <Switch
              checked={toggle.checked}
              className={cn('pointer-events-none shrink-0', !toggle.checked && 'opacity-60')}
              size="xs"
              tabIndex={-1}
            />
          </DropdownMenuItem>
        )}
        {items.map(item => (
          <DropdownMenuItem disabled={item.disabled} key={item.label} onSelect={item.onSelect}>
            {item.label}
          </DropdownMenuItem>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

export function ListStripButton({
  active,
  children,
  disabled,
  onClick
}: {
  active?: boolean
  children: ReactNode
  disabled?: boolean
  onClick: () => void
}) {
  return (
    <button
      className={cn(
        'cursor-pointer text-[0.7rem] font-medium transition-colors disabled:opacity-40',
        active ? 'text-foreground' : 'text-muted-foreground/70 hover:text-foreground'
      )}
      disabled={disabled}
      onClick={onClick}
      type="button"
    >
      {children}
    </button>
  )
}

interface CapRowProps {
  active: boolean
  busy?: boolean
  enabled: boolean
  meta?: ReactNode
  onSelect: () => void
  onToggle: (checked: boolean) => void
  rowId?: string
  /** Second line under the name (category, description, status). Rows grow to h-11. */
  subtitle?: ReactNode
  title: string
  toggleLabel: string
}

// The one row used by all three lists. Fixed height, always-visible switch —
// state reads from the switch + dimmed title, toggling never requires
// selecting first. Off rows dim; the switch itself dims when off.
export function CapRow({
  active,
  busy,
  enabled,
  meta,
  onSelect,
  onToggle,
  rowId,
  subtitle,
  title,
  toggleLabel
}: CapRowProps) {
  return (
    <div
      className={cn(
        'group/row row-hover mb-1 flex w-full shrink-0 items-center rounded-[var(--radius-sm)] border border-transparent transition-[background-color,border-color,box-shadow,color,transform] duration-200 ease-[cubic-bezier(0.2,0.8,0.2,1)] hover:border-(--ui-stroke-quaternary) hover:bg-(--chrome-action-hover) hover:text-foreground',
        subtitle ? 'min-h-12' : 'h-9',
        active
          ? 'border-(--ui-stroke-secondary) bg-(--ui-row-active-background) text-foreground shadow-[0_1px_2px_rgb(0_0_0/0.04)]'
          : 'text-(--ui-text-secondary)'
      )}
      id={rowId}
    >
      <RowButton
        className="flex h-full min-w-0 flex-1 cursor-pointer items-center gap-2 rounded-[var(--radius-sm)] pl-2.5 pr-2 text-left"
        onClick={onSelect}
      >
        <span className="min-w-0 flex-1">
          <span
            className={cn(
              'block truncate text-[0.8125rem]',
              enabled ? 'font-medium text-foreground/85' : 'font-normal text-muted-foreground/60'
            )}
          >
            {title}
          </span>
          {subtitle != null && (
            <span className="mt-0.5 flex min-w-0 items-center gap-1 text-[0.68rem] text-muted-foreground/65">
              {typeof subtitle === 'string' ? <span className="truncate">{subtitle}</span> : subtitle}
            </span>
          )}
        </span>
        {meta != null && (
          <span className="shrink-0 rounded-[var(--radius-sm)] bg-(--ui-bg-quinary) px-1.5 py-0.5 text-[0.62rem] tabular-nums leading-3.5 text-(--ui-text-tertiary)">
            {meta}
          </span>
        )}
      </RowButton>
      <Switch
        aria-label={toggleLabel}
        checked={enabled}
        className={cn('mr-2 shrink-0 cursor-pointer', !enabled && 'opacity-60')}
        disabled={busy}
        onCheckedChange={onToggle}
        size="xs"
        title={toggleLabel}
      />
    </div>
  )
}
