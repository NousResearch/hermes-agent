import * as React from 'react'

import { cn } from '@/lib/utils'

import { type ControlVariantProps, controlVariants } from './control'

function Textarea({ className, size, ...props }: React.ComponentProps<'textarea'> & ControlVariantProps) {
  return <textarea className={cn(controlVariants({ size }), 'min-h-16', className)} data-slot="textarea" {...props} />
}

export { Textarea }
