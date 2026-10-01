import * as React from 'react'

import { cn } from '@/lib/utils'

import { type ControlVariantProps, controlVariants } from './control'

function Textarea({ className, size, variant = 'default', ...props }: React.ComponentProps<'textarea'> & ControlVariantProps & {
  /** Embedded editors let their enclosing surface own the frame and spacing. */
  variant?: 'default' | 'plain'
}) {
  return (
    <textarea
      // Off by default for every consumer — these are code/config/prompt fields,
      // not prose. Callers can re-enable per-instance by passing the prop.
      autoCapitalize="off"
      autoComplete="off"
      autoCorrect="off"
      className={cn(variant === 'plain'
        ? 'w-full min-w-0 border-0 bg-transparent text-foreground shadow-none outline-none placeholder:text-muted-foreground disabled:cursor-not-allowed disabled:opacity-50'
        : controlVariants({ size }), 'min-h-16', className)}
      data-slot="textarea"
      data-variant={variant}
      spellCheck={false}
      {...props}
    />
  )
}

export { Textarea }
