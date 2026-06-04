'use client'

import { type ComponentPropsWithRef, forwardRef } from 'react'

import { Button } from '@/components/ui/button'
import { cn } from '@/lib/utils'

export interface TooltipIconButtonProps extends ComponentPropsWithRef<typeof Button> {
  tooltip: string
  side?: 'top' | 'bottom' | 'left' | 'right'
}

export const TooltipIconButton = forwardRef<HTMLButtonElement, TooltipIconButtonProps>(
  ({ children, tooltip, side: _side = 'bottom', className, ...rest }, ref) => {
    return (
      <Button
        size="icon"
        variant="ghost"
        {...rest}
        aria-label={tooltip}
        className={cn('aui-button-icon size-6 p-1', className)}
        ref={ref}
        title={tooltip}
      >
        {children}
      </Button>
    )
  }
)

TooltipIconButton.displayName = 'TooltipIconButton'
