import type { ComponentProps, ReactNode } from 'react'
import { createContext, useContext } from 'react'

import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Skeleton } from '@/components/ui/skeleton'
import { Switch } from '@/components/ui/switch'
import { triggerHaptic } from '@/lib/haptics'
import type { IconComponent } from '@/lib/icons'
import { cn } from '@/lib/utils'

import { PAGE_INSET_X } from '../layout-constants'

// The settings shell owns page titles; embedded callers retain their headings.
export const SettingsBreadcrumbContext = createContext(false)

// `bare` drops the page gutters + tall bottom pad for embedding in a tighter
// surface (e.g. the boot-failure recovery card owns its own padding).
export function SettingsContent({ children, bare = false }: { children: ReactNode; bare?: boolean }) {
  return (
    <section className="min-h-0 overflow-hidden">
      <div className={cn('h-full min-h-0 overflow-y-auto', bare ? 'px-5 pb-6' : cn('pb-20', PAGE_INSET_X))}>
        {children}
      </div>
    </section>
  )
}

const PILL_VARIANT = {
  muted: 'muted',
  primary: 'default',
  success: 'success',
  warn: 'warn',
  destructive: 'destructive'
} as const

// Rest props spread through to the Badge's DOM node — REQUIRED for Radix
// `asChild` composition (wrapping a Pill in `Tip` clones it with the hover
// handlers and ref as props; swallowing them left every tooltip on a Pill
// silently dead).
export function Pill({
  tone = 'muted',
  children,
  ...props
}: { tone?: keyof typeof PILL_VARIANT; children: ReactNode } & Omit<ComponentProps<typeof Badge>, 'variant'>) {
  return (
    <Badge variant={PILL_VARIANT[tone]} {...props}>
      {children}
    </Badge>
  )
}

export function SectionHeading({
  aside,
  icon: Icon,
  meta,
  page = false,
  title
}: {
  // Right-aligned trailing content on the heading row (e.g. a compact status +
  // action), so a single-item section needn't repeat its own label as a row.
  aside?: ReactNode
  icon: IconComponent
  meta?: string
  page?: boolean
  title: string
}) {
  const hasBreadcrumb = useContext(SettingsBreadcrumbContext)
  const showTitle = !page || !hasBreadcrumb

  if (!showTitle && !aside && !meta) {
    return null
  }

  return (
    <div className="mb-2.5 flex items-center gap-2 pt-2 text-[length:var(--conversation-text-font-size)] font-medium">
      {showTitle && (
        <>
          <Icon className="size-4 shrink-0 text-muted-foreground" />
          <span>{title}</span>
        </>
      )}
      {meta && <Pill>{meta}</Pill>}
      {aside && <div className="ml-auto flex min-w-0 items-center">{aside}</div>}
    </div>
  )
}

// A titled section: heading + body with the shared vertical rhythm. Keeps the
// heading and its content welded together so pages stop hand-rolling
// `<div className="mb-…"><SectionHeading/>…</div>` at every call site.
export function SettingsSection({
  aside,
  children,
  icon,
  meta,
  title
}: {
  aside?: ReactNode
  children: ReactNode
  icon: IconComponent
  meta?: string
  title: string
}) {
  return (
    <section className="mb-6">
      <SectionHeading aside={aside} icon={icon} meta={meta} title={title} />
      {children}
    </section>
  )
}

export function NavLink({
  icon: Icon,
  label,
  active,
  onClick
}: {
  icon: IconComponent
  label: string
  active: boolean
  onClick: () => void
}) {
  return (
    <Button
      className={cn(
        'flex min-h-7 w-full justify-start gap-2 rounded-md px-2 text-left text-[length:var(--conversation-text-font-size)] transition',
        active
          ? 'bg-(--ui-bg-tertiary) text-foreground'
          : 'text-(--ui-text-secondary) hover:bg-(--chrome-action-hover) hover:text-foreground'
      )}
      onClick={onClick}
      size="sm"
      type="button"
      variant="ghost"
    >
      <Icon className="size-4 shrink-0" />
      <span className="min-w-0 flex-1 truncate">{label}</span>
    </Button>
  )
}

// The label/control split every settings-style row keys on. Rows that cannot
// be a ListRow (expandable credential cards, the billing plan card with its
// tier art) still lay out on these exact columns so their controls line up.
export const LIST_ROW_COLUMNS = '@2xl:grid-cols-[minmax(0,1fr)_minmax(15rem,22rem)]'

// The one settings row. `action` is the row's control: beside the label on a
// wide pane, under the description on a narrow one — the row owns that
// alignment, so callers never wrap a control in `justify-end`/`items-end`.
// `wide` rows (galleries, editors) keep the control on the title line and give
// the full width to `below`.
export function ListRow({
  title,
  description,
  hint,
  action,
  below,
  'data-tour': dataTour,
  id,
  wide = false,
  className
}: {
  title: ReactNode
  description?: ReactNode
  hint?: ReactNode
  action?: ReactNode
  below?: ReactNode
  /** Durable handle for tours (see lib/tour) — usually the field's schema key. */
  'data-tour'?: string
  id?: string
  wide?: boolean
  className?: string
}) {
  return (
    // Container-queried, not viewport-queried: the label/control split keys on
    // the row's own pane width, so a narrow detail column (messaging, split
    // views) stacks instead of squishing the label against minmax(15rem,…).
    <div className={cn('@container', className)} data-tour={dataTour} id={id}>
      <div className={cn('grid gap-3 py-3', !wide && [LIST_ROW_COLUMNS, '@2xl:items-center'])}>
        <div className="min-w-0">
          <div className="flex items-center justify-between gap-3 text-[length:var(--conversation-text-font-size)] font-medium text-foreground">
            <span className="min-w-0">{title}</span>
            {wide && action && <div className="flex shrink-0 items-center gap-2">{action}</div>}
          </div>
          {description && (
            <div className="mt-1 text-[length:var(--conversation-caption-font-size)] leading-(--conversation-caption-line-height) text-(--ui-text-tertiary)">
              {description}
            </div>
          )}
          {hint && <div className="mt-1 block font-mono text-[0.68rem] text-muted-foreground/45">{hint}</div>}
          {below}
        </div>
        {!wide && action && (
          <div className="flex min-w-0 flex-wrap items-center gap-2 @2xl:justify-self-end @2xl:justify-end">
            {action}
          </div>
        )}
      </div>
    </div>
  )
}

// The one boolean row: every on/off setting is a Switch in a ListRow (haptic
// baked in). Never a two-option SegmentedControl — that is for choices.
export function ToggleRow({
  checked,
  below,
  'data-tour': dataTour,
  description,
  disabled,
  hint,
  id,
  label,
  onChange,
  wide
}: {
  checked: boolean
  below?: ReactNode
  'data-tour'?: string
  description?: ReactNode
  disabled?: boolean
  hint?: ReactNode
  id?: string
  label: string
  onChange: (on: boolean) => void
  wide?: boolean
}) {
  return (
    <ListRow
      action={
        <Switch
          aria-label={label}
          checked={checked}
          disabled={disabled}
          onCheckedChange={on => {
            triggerHaptic('selection')
            onChange(on)
          }}
        />
      }
      below={below}
      data-tour={dataTour}
      description={description}
      hint={hint}
      id={id}
      title={label}
      wide={wide}
    />
  )
}

// A quiet follow-up under a row's description ("Show 9 tips again", "Reset")
// — the one place a row's secondary action lives, so it never floats beside
// or under the control.
export function RowFootnoteAction({ children, onClick }: { children: ReactNode; onClick: () => void }) {
  return (
    <div className="mt-1.5">
      <Button
        onClick={() => {
          triggerHaptic('selection')
          onClick()
        }}
        size="inline"
        variant="text"
      >
        {children}
      </Button>
    </div>
  )
}

// Skeleton primitives mirroring the settings layout rhythm — a loading page keeps
// its shape (like ModelSettings) instead of collapsing to a centered spinner.
export function SectionHeadingSkeleton() {
  return (
    <div className="mb-2.5 flex items-center gap-2 pt-2">
      <Skeleton className="size-4" />
      <Skeleton className="h-4 w-36 max-w-full" />
    </div>
  )
}

export function ListRowSkeleton({ wide = false }: { wide?: boolean }) {
  return (
    <div className="@container">
      <div className={cn('grid gap-3 py-3', !wide && [LIST_ROW_COLUMNS, '@2xl:items-center'])}>
        <div className="min-w-0 space-y-1.5">
          <Skeleton className="h-3.5 w-40 max-w-full" />
          <Skeleton className="h-3 w-64 max-w-full" />
        </div>
        {!wide && <Skeleton className="h-8 w-full @2xl:w-72 @2xl:justify-self-end" />}
      </div>
    </div>
  )
}

// A full settings page in its loading shape: an optional leading search field
// over one or more sections, each an optional heading above a run of rows.
// `<SettingsSkeleton search sections={[{ heading, rows }]} />`.
export function SettingsSkeleton({
  search = false,
  sections = [{ rows: 4 }]
}: {
  search?: boolean
  sections?: { heading?: boolean; rows: number }[]
}) {
  return (
    <SettingsContent>
      {search && <Skeleton className="mb-3 h-8 w-full" />}
      {sections.map((section, i) => (
        <section className={cn(i > 0 && 'mt-6')} key={i}>
          {section.heading && <SectionHeadingSkeleton />}
          <div className="grid gap-1">
            {Array.from({ length: section.rows }, (_, r) => (
              <ListRowSkeleton key={r} />
            ))}
          </div>
        </section>
      ))}
    </SettingsContent>
  )
}

// Canonical implementation lives in components/ui; re-exported so the many
// settings call sites keep their import path.
export { EmptyState } from '@/components/ui/empty-state'
