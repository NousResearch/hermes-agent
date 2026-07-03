import type { ReactNode } from 'react'

import type { IconComponent } from '@/lib/icons'
import { cn } from '@/lib/utils'

import { PAGE_INSET_X, PAGE_MAX_W } from '../layout-constants'

interface OverlaySplitLayoutProps {
  children: ReactNode
  className?: string
}

interface OverlaySidebarProps {
  children: ReactNode
  className?: string
}

interface OverlayMainProps {
  children: ReactNode
  className?: string
}

interface OverlayNavItemProps {
  active: boolean
  icon: IconComponent
  label: string
  // Renders as an indented child of another nav item: smaller icon and a
  // lighter active state so it never competes with the boxed parent item.
  nested?: boolean
  onClick: () => void
  trailing?: ReactNode
}

export function OverlaySplitLayout({ children, className }: OverlaySplitLayoutProps) {
  return (
    <div
      className={cn(
        'grid h-full min-h-0 flex-1 grid-cols-[13rem_minmax(0,1fr)] overflow-hidden bg-transparent max-[47.5rem]:grid-cols-1',
        className
      )}
    >
      {children}
    </div>
  )
}

export function OverlaySidebar({ children, className }: OverlaySidebarProps) {
  return (
    <aside
      className={cn(
        // pt clears the in-card close button (the OverlayView now insets the
        // whole card below the OS titlebar); the bg fills from the card's top
        // edge so there's no surface-colored gap above the sidebar.
        'flex min-h-0 flex-col gap-0.5 overflow-y-auto bg-(--ui-sidebar-surface-background) px-2.5 pb-3 pt-[calc(var(--titlebar-height)/2+1rem)]',
        className
      )}
    >
      {children}
    </aside>
  )
}

export function OverlayMain({ children, className }: OverlayMainProps) {
  return (
    <main
      className={cn(
        'mx-auto flex min-h-0 w-full flex-1 flex-col overflow-hidden bg-transparent pb-3 pt-[calc(var(--titlebar-height)/2+1rem)]',
        PAGE_MAX_W,
        PAGE_INSET_X,
        className
      )}
    >
      {children}
    </main>
  )
}

export function OverlayNavItem({ active, icon: Icon, label, nested, onClick, trailing }: OverlayNavItemProps) {
  return (
    <button
      className={cn(
        'flex h-7 w-full items-center justify-start gap-2 rounded-md border px-2 text-left text-[length:var(--conversation-text-font-size)] font-normal transition-colors',
        nested
          ? active
            ? 'border-transparent bg-(--chrome-action-hover) font-medium text-foreground'
            : 'border-transparent bg-transparent text-(--ui-text-tertiary) hover:bg-(--chrome-action-hover) hover:text-foreground'
          : active
            ? 'border-(--ui-stroke-tertiary) bg-(--ui-bg-tertiary) text-foreground'
            : 'border-transparent bg-transparent text-(--ui-text-secondary) hover:bg-(--chrome-action-hover) hover:text-foreground'
      )}
      onClick={onClick}
      type="button"
    >
      <Icon
        className={cn(
          'shrink-0',
          nested ? 'size-3.5' : 'size-4',
          active ? 'text-foreground/80' : 'text-muted-foreground/80'
        )}
      />
      <span className="min-w-0 flex-1 truncate">{label}</span>
      {trailing}
    </button>
  )
}
