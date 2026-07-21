import type { ReactNode } from 'react'

import { PageLoader } from '@/components/page-loader'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
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
      <div className={cn('h-full min-h-0 overflow-y-auto', bare ? 'px-5 pb-6' : cn('pb-20', PAGE_INSET_X))}>
        {children}
      </div>
    </section>
  )
}

const PILL_VARIANT = { muted: 'muted', primary: 'default', warn: 'warn' } as const

export function Pill({ tone = 'muted', children }: { tone?: keyof typeof PILL_VARIANT; children: ReactNode }) {
  return <Badge variant={PILL_VARIANT[tone]}>{children}</Badge>
}

export function SectionHeading({ icon: Icon, title, meta }: { icon: IconComponent; title: string; meta?: string }) {
  return (
    <div className="mb-2.5 flex items-center gap-2 pt-2 text-[length:var(--conversation-text-font-size)] font-medium">
      <Icon className="size-4 text-muted-foreground" />
      <span>{title}</span>
      {meta && <Pill>{meta}</Pill>}
    </div>
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
  wide = false
}: {
  title: ReactNode
  description?: ReactNode
  hint?: ReactNode
  action?: ReactNode
  below?: ReactNode
  wide?: boolean
}) {
  return (
    // Container-queried, not viewport-queried: the label/control split keys on
    // the row's own pane width, so a narrow detail column (messaging, split
    // views) stacks instead of squishing the label against minmax(15rem,…).
    <div className="@container">
      <div
        className={cn(
          'grid gap-3 py-3',
          !wide && '@2xl:grid-cols-[minmax(0,1fr)_minmax(15rem,22rem)] @2xl:items-center'
        )}
      >
        <div className="min-w-0">
          <div className="text-[length:var(--conversation-text-font-size)] font-medium text-foreground">{title}</div>
          {description && (
            <div className="mt-1 text-[length:var(--conversation-caption-font-size)] leading-(--conversation-caption-line-height) text-(--ui-text-tertiary)">
              {description}
            </div>
          )}
          {hint && <div className="mt-1 block font-mono text-[0.68rem] text-muted-foreground/45">{hint}</div>}
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

// The settings panels render this as the sole child of the top-padded
// OverlayMain (pt = titlebar + 1rem, no bottom pad — see settings/index.tsx).
// Cancel that top pad so the loader centers in the whole card, not just the
// band beneath it. Inline loaders (mid-panel) should use <PageLoader> directly.
export function LoadingState({ label }: { label: string }) {
  return (
    <PageLoader
      className="-mt-[calc(var(--titlebar-height)+1rem)] h-[calc(100%+var(--titlebar-height)+1rem)]"
      label={label}
    />
  )
}

// Canonical implementation lives in components/ui; re-exported so the many
// settings call sites keep their import path.
export { EmptyState } from '@/components/ui/empty-state'
