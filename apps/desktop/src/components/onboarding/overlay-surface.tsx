import type { ReactNode } from 'react'

import { cn } from '@/lib/utils'

/**
 * The one first-run layer (z 1300). It masks the app but stops above the status bar, where the free
 * account's progress shows (D22). Must stay filled under window glass or the shell shows through:
 * `[data-glass-opaque]` in styles.css.
 */
export function OverlaySurface({
  children,
  className,
  statusbarVisible
}: {
  children: ReactNode
  className?: string
  statusbarVisible: boolean
}) {
  return (
    <div
      className={cn(
        'fixed inset-x-0 top-0 z-(--z-onboarding) flex items-center justify-center bg-(--ui-chat-surface-background) p-6',
        statusbarVisible ? 'bottom-5' : 'bottom-0',
        className
      )}
      data-glass-opaque=""
    >
      {children}
    </div>
  )
}
