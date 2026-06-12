import type { Unstable_TriggerAdapter } from '@assistant-ui/core'
import { ComposerPrimitive } from '@assistant-ui/react'
import type { ReactNode } from 'react'

import { composerFusedDockCard } from '@/components/chat/composer-dock'
import { cn } from '@/lib/utils'

// Same docked chrome as the queue/status stack, but its own thing: a narrow,
// left-aligned card (not full width) that fuses to the composer's edge instead
// of floating above it. `left-1` matches the stack's `mx-1` inset; the negative
// margin overlaps the seam so the composer's (now-transparent) edge border reads
// as shared. Fused (opaque) fill — the composer surface swaps to the same fill
// while a drawer is open, so the two paint as one panel.
const DRAWER_SHELL =
  'absolute left-1 z-50 w-80 max-w-[calc(100%-0.5rem)] max-h-[min(22rem,calc(100vh-8rem))] overflow-y-auto overscroll-contain p-1 text-xs text-popover-foreground'

export const COMPLETION_DRAWER_CLASS = cn(DRAWER_SHELL, 'bottom-full -mb-[9px]', composerFusedDockCard('top'))

export const COMPLETION_DRAWER_BELOW_CLASS = cn(DRAWER_SHELL, 'top-full -mt-[9px]', composerFusedDockCard('bottom'))

export function ComposerCompletionDrawer({
  adapter,
  ariaLabel,
  char,
  children
}: {
  adapter: Unstable_TriggerAdapter
  ariaLabel: string
  char: string
  children: ReactNode
}) {
  return (
    <ComposerPrimitive.Unstable_TriggerPopover
      adapter={adapter}
      aria-label={ariaLabel}
      char={char}
      className={COMPLETION_DRAWER_CLASS}
      data-slot="composer-completion-drawer"
    >
      {children}
    </ComposerPrimitive.Unstable_TriggerPopover>
  )
}

export function CompletionDrawerEmpty({ children, title }: { children?: ReactNode; title: string }) {
  return (
    <div className="px-3 py-3 text-xs text-(--ui-text-tertiary)">
      <p>{title}</p>
      {children && <p className="mt-1 text-xs text-(--ui-text-tertiary)">{children}</p>}
    </div>
  )
}
