'use client'

import { type ComponentPropsWithRef, forwardRef } from 'react'

import { Button } from '@/components/ui/button'
import { Tip } from '@/components/ui/tooltip'
import { cn } from '@/lib/utils'

export interface TooltipIconButtonProps extends ComponentPropsWithRef<typeof Button> {
  tooltip: string
  side?: 'top' | 'bottom' | 'left' | 'right'
}

export const TooltipIconButton = forwardRef<HTMLButtonElement, TooltipIconButtonProps>(
  ({ children, tooltip, side = 'bottom', className, ...rest }, ref) => {
    return (
      <Tip label={tooltip} side={side}>
        <Button
          size="icon-xs"
          variant="ghost"
          {...rest}
          aria-label={tooltip}
          className={cn('aui-button-icon', className)}
          ref={ref}
        >
          {children}
        </Button>
      </Tip>
    )
  }
)

TooltipIconButton.displayName = 'TooltipIconButton'
