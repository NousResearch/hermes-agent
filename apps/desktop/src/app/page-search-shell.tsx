import type { ReactNode } from 'react'

import { cn } from '@/lib/utils'

import { PageSearchInput } from './overlays/overlay-search-input'

interface PageSearchShellProps extends React.ComponentProps<'section'> {
  children: ReactNode
  filters?: ReactNode
  onSearchChange: (value: string) => void
  searchPlaceholder: string
  searchTrailingAction?: ReactNode
  searchValue: string
}

export function PageSearchShell({
  children,
  className,
  filters,
  onSearchChange,
  searchPlaceholder,
  searchTrailingAction,
  searchValue,
  ...props
}: PageSearchShellProps) {
  return (
    <section
      {...props}
      className={cn('flex h-full min-w-0 flex-col overflow-hidden bg-(--ui-chat-surface-background)', className)}
    >
      <div className="relative z-10 grid gap-2 border-b border-(--ui-stroke-tertiary) px-3 py-2.5">
        {/* Reserve the top-right titlebar tools + native window-controls
            footprint so the full-width search input never slides under them
            (this header sits in the titlebar row at the window top). */}
        <div
          style={{
            paddingRight:
              'max(0px, calc(var(--titlebar-tools-right, 0px) + var(--titlebar-tools-width, 0px) - 0.75rem))'
          }}
        >
          <PageSearchInput
            onChange={onSearchChange}
            placeholder={searchPlaceholder}
            trailingAction={searchTrailingAction}
            value={searchValue}
          />
        </div>
        {filters ? <div className="flex flex-wrap items-center justify-center gap-1.5">{filters}</div> : null}
      </div>
      <div className="min-h-0 flex-1 overflow-hidden bg-(--ui-chat-surface-background)">{children}</div>
    </section>
  )
}
