import * as React from 'react'

import { cn } from '@/lib/utils'

import { type ControlVariantProps, controlVariants } from './control'
import { useFieldControl } from './field'

function Textarea({ className, chrome, size, ...props }: React.ComponentProps<'textarea'> & ControlVariantProps) {
  const status = useFieldControl()

  return (
    <textarea
      {...status}
      // Off by default for every consumer — these are code/config/prompt fields,
      // not prose. Callers can re-enable per-instance by passing the prop.
      autoCapitalize="off"
      autoComplete="off"
      autoCorrect="off"
      className={cn(controlVariants({ chrome, size }), 'min-h-16', className)}
      data-slot="textarea"
      spellCheck={false}
      {...props}
    />
  )
}

export { Textarea }
