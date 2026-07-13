import { cva, type VariantProps } from 'class-variance-authority'
import * as React from 'react'

import { cn } from '@/lib/utils'

const alertVariants = cva(
  'relative grid w-full grid-cols-[auto_minmax(0,1fr)] items-start gap-x-3 gap-y-1 rounded-lg border bg-card px-4 py-3 text-sm text-card-foreground shadow-xs [&>svg]:mt-0.5 [&>svg]:size-4 [&>svg]:shrink-0',
  {
    variants: {
      variant: {
        default: 'border-border',
        destructive:
          'border-destructive/35 bg-[color-mix(in_srgb,var(--dt-card)_96%,var(--dt-destructive)_4%)] [&>svg]:text-destructive',
        warning:
          'border-primary/30 bg-[color-mix(in_srgb,var(--dt-card)_96%,var(--dt-primary)_4%)] [&>svg]:text-primary',
        success:
          'border-primary/25 bg-[color-mix(in_srgb,var(--dt-card)_97%,var(--dt-primary)_3%)] [&>svg]:text-primary'
      }
    },
    defaultVariants: {
      variant: 'default'
    }
  }
)

function Alert({ className, variant, ...props }: React.ComponentProps<'div'> & VariantProps<typeof alertVariants>) {
  return <div className={cn(alertVariants({ variant }), className)} data-slot="alert" role="alert" {...props} />
}

function AlertTitle({ className, ...props }: React.ComponentProps<'div'>) {
  return (
    <div
      className={cn('col-start-2 line-clamp-1 min-h-4 font-medium tracking-tight text-foreground', className)}
      data-slot="alert-title"
      {...props}
    />
  )
}

function AlertDescription({ className, ...props }: React.ComponentProps<'div'>) {
  return (
    <div
      className={cn(
        'col-start-2 grid justify-items-start gap-1 text-muted-foreground [&_p]:leading-relaxed',
        className
      )}
      data-slot="alert-description"
      {...props}
    />
  )
}

export { Alert, AlertDescription, AlertTitle }
