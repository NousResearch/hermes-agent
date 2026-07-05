import type { ButtonHTMLAttributes, ReactNode } from 'react'

import { Button } from '@/components/ui/button'
import { cn } from '@/lib/utils'

interface OverlayActionButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  tone?: 'default' | 'danger' | 'subtle'
}

export function OverlayActionButton({
  children,
  className,
  tone = 'default',
  type = 'button',
  ...props
}: OverlayActionButtonProps) {
  return (
    <button
      className={cn(
        'inline-flex h-8 items-center rounded-md border px-3 text-xs font-medium transition-colors disabled:cursor-default disabled:opacity-45',
        tone === 'default' &&
          'border-[color-mix(in_srgb,var(--dt-border)_55%,transparent)] bg-[color-mix(in_srgb,var(--dt-card)_80%,transparent)] text-foreground hover:bg-[color-mix(in_srgb,var(--dt-muted)_46%,var(--dt-card))]',
        tone === 'subtle' &&
          'h-7 border-transparent px-2 text-muted-foreground hover:border-[color-mix(in_srgb,var(--dt-border)_54%,transparent)] hover:bg-[color-mix(in_srgb,var(--dt-card)_72%,transparent)] hover:text-foreground',
        tone === 'danger' &&
          'h-7 border-transparent px-2 text-destructive hover:border-[color-mix(in_srgb,var(--dt-destructive)_40%,transparent)] hover:bg-[color-mix(in_srgb,var(--dt-destructive)_10%,transparent)] hover:text-destructive',
        className
      )}
      type={type}
      {...props}
    >
      {children}
    </button>
  )
}

interface OverlayIconButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  children: ReactNode
}

// Overlay chrome icon action — same titlebar-sized ghost button as the overlay
// close (X), so footer/header actions read identically across breakpoints.
export function OverlayIconButton({ children, className, type = 'button', ...props }: OverlayIconButtonProps) {
  return (
    <Button
      className={cn('text-(--ui-text-tertiary) hover:bg-(--chrome-action-hover) hover:text-foreground', className)}
      size="icon-titlebar"
      type={type}
      variant="ghost"
      {...props}
    >
      {children}
    </Button>
  )
}
