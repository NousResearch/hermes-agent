import { Fragment } from 'react'

import { Codicon } from '@/components/ui/codicon'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger
} from '@/components/ui/dropdown-menu'
import { CountSkeleton } from '@/components/ui/skeleton'
import { TextTab, TextTabMeta } from '@/components/ui/text-tab'
import { compactNumber } from '@/lib/format'
import type { IconComponent } from '@/lib/icons'
import { cn } from '@/lib/utils'

// A count badge beside a tab label. `null` = still loading (pulsing chip, not a
// fake 0); numbers render compact; strings pass through; `undefined` = no badge.
export type TabMeta = number | string | null | undefined

export function tabMetaContent(meta: number | string | null) {
  return meta === null ? <CountSkeleton /> : typeof meta === 'number' ? compactNumber(meta) : meta
}

export interface TabDropdownItem {
  active: boolean
  id: string
  icon?: IconComponent
  /** Indent as a sub-item (flattened nested nav). */
  indent?: boolean
  label: string
  meta?: number | string | null
  onSelect: () => void
  /** Draw a separator above this item (group break). */
  separatorBefore?: boolean
}

function TabDropdownIcon({ icon: Icon, indent }: { icon: IconComponent; indent?: boolean }) {
  return <Icon className={cn('shrink-0 text-muted-foreground/80', indent ? 'size-3.5' : 'size-4')} />
}

/** The Capabilities tab dropdown: a borderless "Label ⌄" trigger and a menu of
 *  labels with right-aligned meta. The single narrow-width collapse used by
 *  every responsive tab/nav in the app. */
export function TabDropdown({
  align = 'center',
  className,
  items
}: {
  align?: 'center' | 'end' | 'start'
  className?: string
  items: TabDropdownItem[]
}) {
  const active = items.find(item => item.active) ?? items[0]

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <button
          className="flex h-7 cursor-pointer items-center gap-1.5 px-1 text-[length:var(--conversation-caption-font-size)] font-medium text-foreground [-webkit-app-region:no-drag]"
          type="button"
        >
          {active?.icon && <TabDropdownIcon icon={active.icon} indent={active.indent} />}
          <span className="min-w-0 truncate">{active?.label}</span>
          {active?.meta !== undefined && <TextTabMeta>{tabMetaContent(active.meta)}</TextTabMeta>}
          <Codicon className="text-muted-foreground" name="chevron-down" size="0.75rem" />
        </button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align={align} className={cn('w-44', className)} sideOffset={6}>
        {items.map((item, index) => (
          <Fragment key={item.id}>
            {item.separatorBefore && index > 0 && <DropdownMenuSeparator />}
            <DropdownMenuItem
              className={cn(item.indent && 'pl-6', item.active && 'text-foreground')}
              onSelect={item.onSelect}
            >
              {item.icon && <TabDropdownIcon icon={item.icon} indent={item.indent} />}
              <span className="min-w-0 flex-1 truncate">{item.label}</span>
              {item.meta !== undefined && (
                <span className="text-xs text-muted-foreground">{tabMetaContent(item.meta)}</span>
              )}
            </DropdownMenuItem>
          </Fragment>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

export interface ResponsiveTab {
  id: string
  label: string
  meta?: number | string | null
}

/** Centered/left `TextTab` row on wide viewports that collapses into a single
 *  `TabDropdown` once the header can't fit it — the shared behavior behind the
 *  Capabilities page tabs, log-source switches, etc. */
export function ResponsiveTabs({
  align = 'center',
  onChange,
  tabs,
  value,
  wideClassName
}: {
  align?: 'center' | 'end' | 'start'
  onChange: (id: string) => void
  tabs: ResponsiveTab[]
  value: string
  /** Extra classes for the wide `TextTab` row (e.g. `justify-center`). */
  wideClassName?: string
}) {
  return (
    <>
      <div className={cn('hidden min-w-0 flex-wrap items-center gap-x-2 gap-y-1 md:flex', wideClassName)}>
        {tabs.map(tab => (
          <TextTab active={tab.id === value} key={tab.id} onClick={() => onChange(tab.id)}>
            {tab.label}
            {tab.meta !== undefined && <TextTabMeta>{tabMetaContent(tab.meta)}</TextTabMeta>}
          </TextTab>
        ))}
      </div>
      <div className="md:hidden">
        <TabDropdown
          align={align}
          items={tabs.map(tab => ({
            active: tab.id === value,
            id: tab.id,
            label: tab.label,
            meta: tab.meta,
            onSelect: () => onChange(tab.id)
          }))}
        />
      </div>
    </>
  )
}
