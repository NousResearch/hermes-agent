import type { ComponentProps } from 'react'

import { Loader } from '@/components/ui/loader'
import { useI18n } from '@/i18n'
import { cn } from '@/lib/utils'

interface PageLoaderProps extends Omit<ComponentProps<'div'>, 'children'> {
  label?: string
}

export function PageLoader({ 'aria-label': ariaLabel, className, label, role = 'status', ...props }: PageLoaderProps) {
  const loadingStatus = useI18n().t.common.loadingStatus

  return (
    <div
      {...props}
      aria-label={ariaLabel ?? label ?? loadingStatus}
      className={cn('grid h-full place-items-center', className)}
      role={role}
    >
      <Loader
        aria-hidden="true"
        className="size-10 text-primary/70"
        pathSteps={220}
        role="presentation"
        strokeScale={0.72}
        type="rose-curve"
      />
    </div>
  )
}
