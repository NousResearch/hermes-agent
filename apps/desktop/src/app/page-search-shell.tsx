import type { ReactNode } from 'react'

import { SearchField } from '@/components/ui/search-field'
import { cn } from '@/lib/utils'

interface PageSearchShellProps extends React.ComponentProps<'section'> {
  children: ReactNode
  /** Primary tabs shown on the top row, beside the search. */
  tabs?: ReactNode
  /** Secondary filters shown full-width on their own row below (expands). */
  filters?: ReactNode
  onSearchChange: (value: string) => void
  searchPlaceholder: string
  searchTrailingAction?: ReactNode
  searchValue: string
  /** Hide the search field when there's nothing to search (empty dataset). */
  searchHidden?: boolean
  /** Reach the underlying input (e.g. to focus it on page mount / hotkey). */
  searchInputRef?: React.RefObject<HTMLInputElement | null>
  /**
   * Centered single-column mode: the header row and the body share one
   * `mx-auto max-w-2xl` column, so the search field lines up (same width, same
   * axis) with the content beneath it — used by the 搜索 page. Off by default,
   * so tab-style pages (插件 / 产物 / 渠道) keep their full-width layout. In this
   * mode the shell owns the scroll container and the centered column; pass just
   * the column's content as `children`.
   */
  centered?: boolean
}

export function PageSearchShell({
  children,
  className,
  tabs,
  filters,
  onSearchChange,
  searchPlaceholder,
  searchTrailingAction,
  searchValue,
  searchHidden = false,
  searchInputRef,
  centered = false,
  ...props
}: PageSearchShellProps) {
  return (
    <section
      {...props}
      className={cn('flex h-full min-w-0 flex-col overflow-hidden bg-(--ui-chat-surface-background)', className)}
    >
      {/*
        Header lives in the page body, below the window chrome (the shell floats
        traffic lights over the top titlebar-height strip, which the `pt` clears
        and leaves draggable). Top row: primary tabs + search. Second row:
        secondary filters, full-width so they expand. Interactive bits opt out
        of the drag region.
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
        {(tabs || !searchHidden) && (
          <div
            className={cn(
              'flex items-center gap-3 pb-2 pt-[calc(var(--titlebar-height)+0.5rem)]',
              // Centered mode shares the body's `mx-auto max-w-2xl px-3` box so the
              // search field lines up with the column beneath it; default keeps the
              // original full-width `px-3` header.
              centered ? 'mx-auto w-full max-w-2xl px-3' : 'px-3'
            )}
          >
            {tabs ? <div className="flex min-w-0 flex-1 flex-wrap items-center gap-x-2 gap-y-1">{tabs}</div> : null}
            {!searchHidden && (
              <div className={cn('flex items-center', centered ? 'flex-1' : cn('shrink-0', !tabs && 'flex-1'))}>
                <SearchField
                  containerClassName={centered ? 'w-full' : 'max-w-[45vw]'}
                  inputRef={searchInputRef}
                  onChange={onSearchChange}
                  placeholder={searchPlaceholder}
                  trailingAction={searchTrailingAction}
                  value={searchValue}
                />
              </div>
            )}
          </div>
        )}
        {filters ? <div className="flex flex-wrap items-center gap-x-2 gap-y-1 px-3 pb-2">{filters}</div> : null}
      </div>
      <div className="min-h-0 flex-1 overflow-hidden bg-(--ui-chat-surface-background)">
        {centered ? (
          // Scroll at the window edge, content in the same centered column as the
          // header. `px-3` matches the header so both left/right edges align.
          <div className="h-full overflow-y-auto">
            <div className="mx-auto w-full max-w-2xl px-3">{children}</div>
          </div>
        ) : (
          children
        )}
      </div>
    </section>
  )
}
