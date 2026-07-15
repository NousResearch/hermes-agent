import { cva, type VariantProps } from 'class-variance-authority'
import { Slot } from 'radix-ui'
import type * as React from 'react'

import { cn } from '@/lib/utils'

// Small status/metadata tag. App radius (not a full pill); tones map to the
// shared accent/muted/destructive surfaces so badges read consistently.
const badgeVariants = cva(
  'inline-flex w-fit shrink-0 items-center gap-1 rounded-[3px] font-medium leading-none whitespace-nowrap [&_svg]:pointer-events-none',
  {
    variants: {
      variant: {
        default: 'bg-primary/10 text-primary',
        muted: 'bg-muted text-muted-foreground',
        warn: 'bg-amber-500/10 text-amber-600 dark:text-amber-300',
        destructive: 'bg-destructive/10 text-destructive',
        outline: 'border border-(--ui-stroke-secondary) text-muted-foreground'
      },
      size: {
        default: 'px-1.5 py-0.5 text-[0.65rem] [&_svg]:size-3',
        xs: 'px-1 py-px text-[0.6rem] [&_svg]:size-2.5'
      }
    },
    defaultVariants: { variant: 'default', size: 'default' }
  }
)

export interface BadgeProps extends React.ComponentProps<'span'>, VariantProps<typeof badgeVariants> {
  asChild?: boolean
}

export function Badge({ asChild = false, className, size, variant, ...props }: BadgeProps) {
  const Comp = asChild ? Slot.Root : 'span'

  return <Comp className={cn(badgeVariants({ size, variant }), className)} data-slot="badge" {...props} />
}

export { badgeVariants }
