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

type PageSearchContentWidth = 'reading' | 'wide'

interface PageSearchShellProps extends Omit<React.ComponentProps<'section'>, 'title'> {
  children: ReactNode
  /** Optional page identity above the search and tab controls. */
  description?: ReactNode
  tabs?: PageShellTab[]
  activeTab?: string
  onTabChange?: (id: string) => void
  title?: ReactNode
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
  /** Centers prose-like catalog pages more tightly than operational wide views. */
  contentWidth?: PageSearchContentWidth
  /** Places the search field on its own readable row below a page introduction. */
  searchBelowTitle?: boolean
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
  title,
  description,
  filters,
  onSearchChange,
  searchPlaceholder,
  searchHints,
  searchValue,
  searchHidden = false,
  searchTrailingAction,
  contentWidth = 'wide',
  searchBelowTitle = false,
  ...props
}: PageSearchShellProps) {
  const hasTabs = (tabs?.length ?? 0) > 0
  const shellWidth = contentWidth === 'reading' ? 'max-w-[52rem]' : 'max-w-[75rem]'
  const shellGutter = contentWidth === 'reading' ? 'px-5' : 'px-[clamp(1.25rem,4vw,4rem)]'
  const hasInlineControls = hasTabs || (!searchHidden && !searchBelowTitle)

  return (
    <section
      {...props}
      className={cn('flex h-full min-w-0 flex-col overflow-hidden bg-(--ui-chat-surface-background)', className)}
    >
      {/*
        Header lives in the page body, below the window chrome (the shell floats
        traffic lights over the top titlebar-height strip, which the `pt` clears
        and leaves draggable). Catalog pages can place a broad search row below
        the intro; operational pages retain the compact search/tabs grid.
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
      <div className="shrink-0 border-b border-(--ui-stroke-quaternary) bg-(--ui-chat-surface-background)">
        {title && (
          <div
            className={cn(
              'mx-auto w-full pt-[calc(var(--titlebar-height)+0.875rem)]',
              shellGutter,
              shellWidth,
              searchBelowTitle ? 'pb-3' : 'pb-4'
            )}
          >
            <h1 className="text-[1.375rem] leading-tight font-semibold tracking-tight text-foreground">{title}</h1>
            {description && (
              <p className="mt-1 text-[0.8125rem] leading-relaxed text-(--ui-text-tertiary)">{description}</p>
            )}
          </div>
        )}
        {searchBelowTitle && !searchHidden && (
          <div className={cn('mx-auto w-full pb-3', shellGutter, shellWidth)}>
            <SearchField
              containerClassName="w-full border-(--ui-stroke-secondary) bg-[color-mix(in_srgb,var(--ui-bg-elevated)_78%,transparent)] px-3 py-0.5 opacity-100 shadow-[0_1px_2px_rgb(0_0_0/0.04)]"
              hints={searchHints}
              inputClassName="h-8 w-full text-[0.8125rem] [field-sizing:auto]"
              onChange={onSearchChange}
              placeholder={searchPlaceholder}
              value={searchValue}
            />
          </div>
        )}
        {hasInlineControls && (
          <div
            className={cn(
              'mx-auto grid w-full grid-cols-[minmax(0,1fr)_auto_minmax(0,1fr)] items-center gap-4 pb-3',
              shellGutter,
              shellWidth,
              title ? 'pt-0' : 'pt-[calc(var(--titlebar-height)+0.75rem)]'
            )}
          >
            <div className="flex min-w-0 items-center justify-start">
              {!searchHidden && !searchBelowTitle && (
                <SearchField
                  containerClassName="w-full max-w-[32rem]"
                  hints={searchHints}
                  inputClassName="w-full [field-sizing:auto]"
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
