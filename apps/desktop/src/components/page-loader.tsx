import type { ComponentProps } from 'react'

import { Loader } from '@/components/ui/loader'
// classify the region wait without replacing Hermes's loader primitive.
import { type SystemActivityProps, SystemActivitySlot } from '@/lib/system-activity'
import { cn } from '@/lib/utils'

interface PageLoaderProps extends Omit<ComponentProps<'div'>, 'children'> {
  label?: string
  activity?: SystemActivityProps['activity']
}

export function PageLoader({
  'aria-label': ariaLabel,
  activity = 'loading',
  className,
  label = 'Loading',
  role = 'status',
  ...props
}: PageLoaderProps) {
  return (
    <div
      {...props}
      aria-label={ariaLabel ?? label}
      className={cn('grid h-full place-items-center', className)}
      role={role}
    >
      <SystemActivitySlot
        activity={activity}
        fallback={
          <Loader
            aria-hidden="true"
            className="size-10 text-primary/70"
            pathSteps={220}
            role="presentation"
            strokeScale={0.72}
            type="rose-curve"
          />
        }
        placement="region"
      />
    </div>
  )
}
