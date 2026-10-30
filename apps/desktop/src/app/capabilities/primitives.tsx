import type { ReactNode } from 'react'

import { Switch } from '@/components/ui/switch'
import { useI18n } from '@/i18n'

import { ListStripButton } from '../master-detail'
import { PanelEmpty } from '../overlays/panel'

// The chrome the Skills and Tools tabs share: one inspector header, one empty
// state and one sort control, so switching between them never jumps.

// Section header for a grouped capability list: the group label, its row
// count, and one group-wide switch. Mixed state reads as off; flipping it on
// enables the whole group (the strip-menu bulk convention, inline).
export function GroupHeaderRow({
  busy,
  count,
  enabled,
  label,
  onToggle
}: {
  busy?: boolean
  count: number
  enabled: boolean
  label: string
  onToggle: (checked: boolean) => void
}) {
  return (
    <div className="flex h-8 shrink-0 items-center gap-1.5 pr-1.5 pl-2">
      <span className="truncate text-[0.72rem] font-semibold text-foreground/85">{label}</span>
      <span className="shrink-0 text-[0.62rem] tabular-nums text-(--ui-text-quaternary)">{count}</span>
      <Switch
        aria-label={label}
        checked={enabled}
        className="ml-auto shrink-0 cursor-pointer"
        disabled={busy}
        onCheckedChange={onToggle}
        title={label}
      />
    </div>
  )
}

// Shared inspector header — mirrors Messaging's PlatformDetail so Skills and
// Tools share one title/description block and tab switches don't jump.
export function DetailHeader({
  description,
  pills,
  title
}: {
  description: ReactNode
  pills?: ReactNode
  title: string
}) {
  return (
    <header>
      <div className="flex min-h-6 flex-wrap items-center gap-2">
        <h3 className="min-w-0 truncate text-[0.9375rem] font-semibold tracking-tight">{title}</h3>
        {pills}
      </div>
      <p className="mt-1 text-[length:var(--conversation-caption-font-size)] leading-(--conversation-caption-line-height) text-(--ui-text-tertiary)">
        {description}
      </p>
    </header>
  )
}

// Full-bleed empty state, matching the MCP tab (spans both columns, not a
// cramped note in the left rail). Query-aware, and says "tools" not the
// internal "toolsets".
export function CapabilityEmpty({ noun, query }: { noun: string; query: string }) {
  const { t } = useI18n()
  const q = query.trim()

  return (
    <div className="flex h-full min-h-0 flex-1">
      <PanelEmpty
        description={q ? t.skills.emptyNothingMatches(q) : t.skills.emptyNoneAvailable(noun)}
        icon="search"
        title={t.skills.emptyNoneFound(noun)}
      />
    </div>
  )
}

/** Most-used / least-used flip for a list strip. */
export function SortButton({ desc, onFlip }: { desc: boolean; onFlip: () => void }) {
  const { t } = useI18n()

  return (
    <ListStripButton onClick={onFlip}>{desc ? t.skills.sortMostUsedDesc : t.skills.sortLeastUsedAsc}</ListStripButton>
  )
}
