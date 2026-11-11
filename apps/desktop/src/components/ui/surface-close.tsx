import type { ComponentProps, ReactNode } from 'react'

import { Button } from '@/components/ui/button'
import { useI18n } from '@/i18n'
import { X } from '@/lib/icons'
import { cn } from '@/lib/utils'

// Always in flow: the parent flex/grid reserves the hit target, never an overlay.
export function SurfaceCloseButton({
  'aria-label': label,
  className,
  ...props
}: Omit<ComponentProps<typeof Button>, 'children' | 'asChild'>) {
  const { t } = useI18n()

  return (
    <Button
      aria-label={label ?? t.common.close}
      className={cn('shrink-0 text-(--ui-text-tertiary) hover:text-foreground [-webkit-app-region:no-drag]', className)}
      data-slot="surface-close"
      size="icon-xs"
      type="button"
      variant="ghost"
      {...props}
    >
      <X aria-hidden className="size-4" />
    </Button>
  )
}

// The entire row is reserved above the body, including search and other actions.
// Grid follows inherited direction: the close slot is left in RTL, right in LTR.
export function SurfaceCloseHeader({
  children,
  close,
  className,
  ...props
}: ComponentProps<'div'> & { close: ReactNode }) {
  return (
    <div
      className={cn('grid min-w-0 shrink-0 grid-cols-[minmax(0,1fr)_auto] items-start gap-2 p-2.5', className)}
      data-slot="surface-close-header"
      {...props}
    >
      <div className="flex min-w-0 flex-wrap items-center gap-2 [overflow-wrap:anywhere] [-webkit-app-region:no-drag] [&>*]:min-w-0 [&>*]:max-w-full [&_button]:whitespace-normal">
        {children}
      </div>
      {close}
    </div>
  )
}
