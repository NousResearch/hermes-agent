import type { ReactNode } from 'react'

import { SearchField } from '@/components/ui/search-field'
import { ResponsiveTabs } from '@/components/ui/tab-dropdown'
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
  /** Right-aligned control in the header's trailing cell (e.g. a refresh button)
   *  so mouse users get a visible affordance for the refresh hotkey. */
  searchTrailingAction?: ReactNode
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
  return (
    <ResponsiveTabs
      onChange={id => onTabChange?.(id)}
      tabs={tabs}
      value={activeTab ?? tabs[0]?.id ?? ''}
      wideClassName="justify-center"
    />
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
  searchTrailingAction,
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
            <div className="flex min-w-0 items-center justify-end">{searchTrailingAction}</div>
          </div>
        )}
        {filters ? <div className="flex flex-wrap items-center gap-x-2 gap-y-1 px-3 pb-2">{filters}</div> : null}
      </div>
      <div className="min-h-0 flex-1 overflow-hidden bg-(--ui-chat-surface-background)">{children}</div>
    </section>
  )
}
