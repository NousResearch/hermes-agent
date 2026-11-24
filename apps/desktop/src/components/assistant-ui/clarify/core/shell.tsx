'use client'

import { useStore } from '@nanostores/react'
import type { ComponentProps } from 'react'

import { WIDGET_SHELL_CLASS } from '@/components/chat/widget-shell'
import type { MessageQuestion } from '@/lib/icons'
import { cn } from '@/lib/utils'
import { $clarifyCardWidth, type ClarifyCardWidth } from '@/store/clarify-card-width'

// field-sizing on top of Textarea's shared chrome; kill min-h-16 for one-liners.
export const CLARIFY_TEXTAREA_CLASS = 'field-sizing-content max-h-40 min-h-0 resize-none'

export const CLARIFY_SHELL_CLASS = `${WIDGET_SHELL_CLASS} text-[length:var(--conversation-text-font-size)] text-(--ui-text-primary)`

export const CLARIFY_ICON_CLASS = 'mt-px size-4 shrink-0 text-(--ui-text-tertiary)'

// Explicit width, so the card never shrink-wraps to its content: `full` matches
// the assistant's prose column, `compact` keeps the narrow reading width.
const CLARIFY_WIDTH_CLASS: Record<ClarifyCardWidth, string> = {
  compact: 'w-full max-w-md',
  full: 'w-full max-w-full'
}

export function useClarifyWidthClass(): string {
  return CLARIFY_WIDTH_CLASS[useStore($clarifyCardWidth)]
}

export function ClarifyShell({ children, className, ...props }: ComponentProps<'div'>) {
  const widthClass = useClarifyWidthClass()

  return (
    <div className={cn(CLARIFY_SHELL_CLASS, widthClass, className)} data-slot="clarify-inline" {...props}>
      {children}
    </div>
  )
}

export function ClarifyLine({
  children,
  className,
  icon: Icon,
  ...props
}: ComponentProps<'div'> & { icon: typeof MessageQuestion }) {
  return (
    <div className={cn('flex items-start gap-2', className)} {...props}>
      <div className="min-w-0 flex-1">{children}</div>
      <Icon aria-hidden className={CLARIFY_ICON_CLASS} />
    </div>
  )
}
