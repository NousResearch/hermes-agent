import { cn } from '@/lib/utils'

function Skeleton({ className, ...props }: React.ComponentProps<'div'>) {
  return <div className={cn('animate-pulse rounded-md bg-accent', className)} data-slot="skeleton" {...props} />
}

/** Inline pulsing chip standing in for a small count/badge while it loads. */
function CountSkeleton({ className, ...props }: React.ComponentProps<'span'>) {
  return (
    <span
      className={cn('inline-block h-2 w-3.5 translate-y-px animate-pulse rounded-sm bg-current/25', className)}
      data-slot="count-skeleton"
      {...props}
    />
  )
}

export { CountSkeleton, Skeleton }
