import type { Unstable_TriggerAdapter } from '@assistant-ui/core'
import { ComposerPrimitive } from '@assistant-ui/react'
import type { ReactNode } from 'react'

export const COMPLETION_DRAWER_CLASS = [
  'absolute bottom-[calc(100%+0.375rem)] left-0 z-50',
  'w-80 max-w-[calc(100vw-2rem)]',
  'max-h-[min(22rem,calc(100vh-8rem))] overflow-y-auto overscroll-contain',
  'rounded-xl border border-(--ui-stroke-secondary)',
  'bg-[color-mix(in_srgb,var(--ui-bg-elevated)_97%,transparent)]',
  'p-1 text-xs text-popover-foreground shadow-lg',
  'backdrop-blur-md'
].join(' ')

export const COMPLETION_DRAWER_BELOW_CLASS = [
  'absolute left-0 top-[calc(100%+0.375rem)] z-50',
  'w-80 max-w-[calc(100vw-2rem)]',
  'max-h-[min(22rem,calc(100vh-8rem))] overflow-y-auto overscroll-contain',
  'rounded-xl border border-(--ui-stroke-secondary)',
  'bg-[color-mix(in_srgb,var(--ui-bg-elevated)_97%,transparent)]',
  'p-1 text-xs text-popover-foreground shadow-lg',
  'backdrop-blur-md'
].join(' ')

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
