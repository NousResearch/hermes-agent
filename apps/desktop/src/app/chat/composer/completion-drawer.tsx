import type { Unstable_TriggerAdapter } from '@assistant-ui/core'
import { ComposerPrimitive } from '@assistant-ui/react'
import type { ReactNode } from 'react'

export const COMPLETION_DRAWER_CLASS = [
  'absolute bottom-[calc(100%+0.25rem)] left-0 z-50',
  'w-60 max-w-[calc(100vw-2rem)]',
  'max-h-[min(23rem,calc(100vh-8rem))] overflow-y-auto overscroll-contain',
  'rounded-lg border border-(--ui-stroke-secondary)',
  'bg-[color-mix(in_srgb,var(--ui-bg-elevated)_96%,transparent)]',
  'p-1 text-xs text-popover-foreground shadow-md',
  'backdrop-blur-md'
].join(' ')

export const COMPLETION_DRAWER_BELOW_CLASS = [
  'absolute left-0 top-[calc(100%+0.25rem)] z-50',
  'w-60 max-w-[calc(100vw-2rem)]',
  'max-h-[min(23rem,calc(100vh-8rem))] overflow-y-auto overscroll-contain',
  'rounded-lg border border-(--ui-stroke-secondary)',
  'bg-[color-mix(in_srgb,var(--ui-bg-elevated)_96%,transparent)]',
  'p-1 text-xs text-popover-foreground shadow-md',
  'backdrop-blur-md'
].join(' ')

export const COMPLETION_DRAWER_ROW_CLASS = [
  'relative flex cursor-default select-none items-center gap-2 rounded-md px-2 py-1',
  'w-full min-w-0 text-left text-xs outline-hidden transition-colors',
  'hover:bg-(--ui-bg-tertiary)',
  'data-[highlighted]:bg-(--ui-bg-tertiary) data-[highlighted]:text-foreground'
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
