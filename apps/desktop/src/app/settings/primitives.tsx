import type { ReactNode } from 'react'

import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Skeleton } from '@/components/ui/skeleton'
import { Switch } from '@/components/ui/switch'
import { triggerHaptic } from '@/lib/haptics'
import type { IconComponent } from '@/lib/icons'
import { cn } from '@/lib/utils'

import { PAGE_INSET_X } from '../layout-constants'

// `bare` drops the page gutters + tall bottom pad for embedding in a tighter
// surface (e.g. the boot-failure recovery card owns its own padding).
export function SettingsContent({ children, bare = false }: { children: ReactNode; bare?: boolean }) {
  return (
    <section className="min-h-0 overflow-hidden">
      <div className="h-full min-h-0 overflow-y-auto">
        <div className={cn('mx-auto w-full', bare ? 'px-5 pb-6' : cn('max-w-[52rem] pb-20 pt-6', PAGE_INSET_X))}>
          {children}
        </div>
      </div>
    </section>
  )
}

/** A quiet, bounded page introduction that makes settings pages scannable. */
export function SettingsPageHeader({ description, title }: { description?: ReactNode; title: ReactNode }) {
  return (
    <header className="mb-5">
      <h1 className="text-lg font-semibold tracking-tight text-foreground">{title}</h1>
      {description && (
        <p className="mt-1.5 max-w-2xl text-[length:var(--conversation-caption-font-size)] leading-(--conversation-caption-line-height) text-(--ui-text-tertiary)">
          {description}
        </p>
      )}
    </header>
  )
}

/** A visually unified set of settings rows, with one surface and clear dividers. */
export function SettingsGroup({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <div
      className={cn(
        'divide-y divide-(--ui-stroke-quaternary) overflow-hidden rounded-2xl border border-(--ui-stroke-secondary) bg-[color-mix(in_srgb,var(--ui-bg-elevated)_72%,transparent)] shadow-[0_1px_2px_rgb(0_0_0/0.04)]',
        className
      )}
    >
      {children}
    </div>
  )
}

const PILL_VARIANT = { muted: 'muted', primary: 'default', warn: 'warn' } as const

export function Pill({ tone = 'muted', children }: { tone?: keyof typeof PILL_VARIANT; children: ReactNode }) {
  return <Badge variant={PILL_VARIANT[tone]}>{children}</Badge>
}

export function SectionHeading({
  aside,
  icon: Icon,
  meta,
  title
}: {
  // Right-aligned trailing content on the heading row (e.g. a compact status +
  // action), so a single-item section needn't repeat its own label as a row.
  aside?: ReactNode
  icon: IconComponent
  meta?: string
  title: string
}) {
  return (
    <div className="mb-2.5 flex items-center gap-2 pt-2 text-[length:var(--conversation-text-font-size)] font-medium">
      <Icon className="size-4 shrink-0 text-muted-foreground" />
      <span>{title}</span>
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

export function ListRow({
  title,
  description,
  hint,
  action,
  below,
  wide = false,
  className
}: {
  title: ReactNode
  description?: ReactNode
  hint?: ReactNode
  action?: ReactNode
  below?: ReactNode
  wide?: boolean
  className?: string
}) {
  return (
    // Container-queried, not viewport-queried: the label/control split keys on
    // the row's own pane width, so a narrow detail column (messaging, split
    // views) stacks instead of squishing the label against minmax(15rem,…).
    <div className={cn('@container px-4 sm:px-5', className)} data-slot="settings-row">
      <div
        className={cn(
          'grid gap-3 py-3.5',
          !wide && '@2xl:grid-cols-[minmax(0,1fr)_minmax(11rem,18rem)] @2xl:items-center'
        )}
      >
        <div className="min-w-0">
          <div className="text-[0.8125rem] font-semibold text-foreground">{title}</div>
          {description && (
            <div className="mt-1 text-[0.75rem] leading-[1.35] text-(--ui-text-tertiary)">{description}</div>
          )}
          {hint && <div className="mt-1.5 block font-mono text-[0.68rem] text-muted-foreground/55">{hint}</div>}
          {below}
        </div>
        {action && <div className={cn('min-w-0', !wide && '@2xl:justify-self-end')}>{action}</div>}
      </div>
    </div>
  )
}

// A labelled on/off row — the canonical device-pref switch (haptic baked in).
export function ToggleRow({
  checked,
  description,
  disabled,
  label,
  onChange
}: {
  checked: boolean
  description?: string
  disabled?: boolean
  label: string
  onChange: (on: boolean) => void
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
      description={description}
      title={label}
    />
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
    <div className="@container px-4 sm:px-5">
      <div
        className={cn(
          'grid gap-3 py-3.5',
          !wide && '@2xl:grid-cols-[minmax(0,1fr)_minmax(11rem,18rem)] @2xl:items-center'
        )}
      >
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
          <SettingsGroup className="grid gap-0">
            {Array.from({ length: section.rows }, (_, r) => (
              <ListRowSkeleton key={r} />
            ))}
          </SettingsGroup>
        </section>
      ))}
    </SettingsContent>
  )
}

// Canonical implementation lives in components/ui; re-exported so the many
// settings call sites keep their import path.
export { EmptyState } from '@/components/ui/empty-state'
