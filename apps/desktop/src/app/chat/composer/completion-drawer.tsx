import type { Unstable_TriggerAdapter } from '@assistant-ui/core'
import { ComposerPrimitive } from '@assistant-ui/react'
import type { ReactNode } from 'react'

import { composerPanelCard } from '@/components/chat/composer-dock'
import { cn } from '@/lib/utils'

// A standalone glassy panel floating just off the composer edge, inset from the
// left. Skin is the shared composerPanelCard (also used by the attach menu).
const DRAWER_SHELL = cn(
  'absolute left-2 z-50 w-80 max-w-[calc(100%-1rem)] max-h-[min(22rem,calc(100vh-8rem))]',
  'overflow-y-auto overscroll-contain p-1 text-popover-foreground',
  composerPanelCard
)

export const COMPLETION_DRAWER_CLASS = cn(DRAWER_SHELL, 'bottom-full mb-1')

export const COMPLETION_DRAWER_BELOW_CLASS = cn(DRAWER_SHELL, 'top-full mt-1')

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
