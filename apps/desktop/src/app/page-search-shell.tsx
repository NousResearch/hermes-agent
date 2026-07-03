import type { ReactNode } from 'react'

import { Codicon } from '@/components/ui/codicon'
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from '@/components/ui/dropdown-menu'
import { SearchField } from '@/components/ui/search-field'
import { CountSkeleton } from '@/components/ui/skeleton'
import { TextTab, TextTabMeta } from '@/components/ui/text-tab'
import { compactNumber } from '@/lib/format'
import { cn } from '@/lib/utils'

// Tabs are data, not nodes: the shell owns their presentation so every page
// gets the same behavior — a centered TextTab row on wide viewports that
// collapses into a dropdown when the header can't fit both search and tabs.
export interface PageShellTab {
  id: string
  label: string
  /** Count badge. `null` = still loading (renders a skeleton); `undefined` = no badge. */
  meta?: string | number | null
}

// null = loading (pulsing chip instead of a fake 0); numbers render compact.
const metaContent = (meta: string | number | null) =>
  meta === null ? <CountSkeleton /> : typeof meta === 'number' ? compactNumber(meta) : meta

interface PageSearchShellProps extends React.ComponentProps<'section'> {
  children: ReactNode
  tabs?: PageShellTab[]
  activeTab?: string
  onTabChange?: (id: string) => void
  /** Secondary filters shown full-width on their own row below (expands). */
  filters?: ReactNode
  onSearchChange: (value: string) => void
  searchPlaceholder: string
  /** Data-derived rotating placeholder nudges (see SearchField.hints). */
  searchHints?: string[]
  searchValue: string
  /** Hide the search field when there's nothing to search (empty dataset). */
  searchHidden?: boolean
}

function ShellTabs({
  tabs,
  activeTab,
  onTabChange
}: {
  tabs: PageShellTab[]
  activeTab?: string
  onTabChange?: (id: string) => void
}) {
  const active = tabs.find(tab => tab.id === activeTab) ?? tabs[0]

  return (
    <>
      <div className="hidden min-w-0 flex-wrap items-center justify-center gap-x-2 gap-y-1 md:flex">
        {tabs.map(tab => (
          <TextTab active={tab.id === activeTab} key={tab.id} onClick={() => onTabChange?.(tab.id)}>
            {tab.label}
            {/* Direct TextTabMeta child — TextTab type-checks for it to keep the
                count outside the active-underline span. */}
            {tab.meta !== undefined && <TextTabMeta>{metaContent(tab.meta)}</TextTabMeta>}
          </TextTab>
        ))}
      </div>
      <div className="md:hidden">
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <button
              className="flex h-7 cursor-pointer items-center gap-1 px-1 text-[length:var(--conversation-caption-font-size)] font-medium text-foreground [-webkit-app-region:no-drag]"
              type="button"
            >
              {active.label}
              {active.meta !== undefined && <TextTabMeta>{metaContent(active.meta)}</TextTabMeta>}
              <Codicon className="text-muted-foreground" name="chevron-down" size="0.75rem" />
            </button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="center" className="w-44" sideOffset={6}>
            {tabs.map(tab => (
              <DropdownMenuItem key={tab.id} onSelect={() => onTabChange?.(tab.id)}>
                <span className="min-w-0 flex-1 truncate">{tab.label}</span>
                {tab.meta !== undefined && (
                  <span className="text-xs text-muted-foreground">{metaContent(tab.meta)}</span>
                )}
              </DropdownMenuItem>
            ))}
          </DropdownMenuContent>
        </DropdownMenu>
      </div>
    </>
  )
}

export function PageSearchShell({
  children,
  className,
  tabs,
  activeTab,
  onTabChange,
  filters,
  onSearchChange,
  searchPlaceholder,
  searchHints,
  searchValue,
  searchHidden = false,
  ...props
}: PageSearchShellProps) {
  const hasTabs = (tabs?.length ?? 0) > 0

  return (
    <section
      {...props}
      className={cn('flex h-full min-w-0 flex-col overflow-hidden bg-(--ui-chat-surface-background)', className)}
    >
      {/*
        Header lives in the page body, below the window chrome (the shell floats
        traffic lights over the top titlebar-height strip, which the `pt` clears
        and leaves draggable). Search left, tabs centered on the page via the
        1fr/auto/1fr grid; the trailing 1fr keeps the center honest.
      */}
      {/*
        IMPORTANT: do NOT put `-webkit-app-region: drag` on this header. It spans
        full width over the band where the floating titlebar icon clusters live,
        and an overlapping OS drag region eats their clicks at the compositor
        level (pointer-events / no-drag carve-outs across separate stacking
        contexts don't reliably fix it on macOS). The shell already supplies a
        draggable titlebar strip that is `calc()`'d around the icon clusters
        (see app-shell.tsx), so window dragging still works here.
      */}
      <div className="shrink-0">
        {(hasTabs || !searchHidden) && (
          <div className="grid grid-cols-[minmax(0,1fr)_auto_minmax(0,1fr)] items-center gap-3 px-3 pb-2 pt-[calc(var(--titlebar-height)+0.5rem)]">
            <div className="flex min-w-0 items-center justify-start">
              {!searchHidden && (
                <SearchField
                  containerClassName="max-w-[45vw]"
                  hints={searchHints}
                  onChange={onSearchChange}
                  placeholder={searchPlaceholder}
                  value={searchValue}
                />
              )}
            </div>
            {hasTabs ? <ShellTabs activeTab={activeTab} onTabChange={onTabChange} tabs={tabs!} /> : <span />}
            <span />
          </div>
        )}
        {filters ? <div className="flex flex-wrap items-center gap-x-2 gap-y-1 px-3 pb-2">{filters}</div> : null}
      </div>
      <div className="min-h-0 flex-1 overflow-hidden bg-(--ui-chat-surface-background)">{children}</div>
    </section>
  )
}
