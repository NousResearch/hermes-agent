import { Tooltip as TooltipPrimitive } from 'radix-ui'
import * as React from 'react'

import { cn } from '@/lib/utils'

function TooltipProvider({ delayDuration = 0, ...props }: React.ComponentProps<typeof TooltipPrimitive.Provider>) {
  return <TooltipPrimitive.Provider data-slot="tooltip-provider" delayDuration={delayDuration} {...props} />
}

function Tooltip({ ...props }: React.ComponentProps<typeof TooltipPrimitive.Root>) {
  return <TooltipPrimitive.Root data-slot="tooltip" {...props} />
}

function TooltipTrigger({ ...props }: React.ComponentProps<typeof TooltipPrimitive.Trigger>) {
  return <TooltipPrimitive.Trigger data-slot="tooltip-trigger" {...props} />
}

function TooltipContent({
  className,
  sideOffset = 6,
  children,
  ...props
}: React.ComponentProps<typeof TooltipPrimitive.Content>) {
  return (
    <TooltipPrimitive.Portal>
      <TooltipPrimitive.Content
        // Instant, no transition (the Provider's delayDuration=0 + no animate-*
        // classes). bg-foreground/text-background auto-inverts per theme: white
        // on near-black in light mode, black on white in dark.
        className={cn(
          'z-[200] w-fit bg-foreground px-1.5 py-1 text-[11px] font-bold leading-none text-background select-none [font-family:Arial,sans-serif]',
          className
        )}
        data-slot="tooltip-content"
        sideOffset={sideOffset}
        {...props}
      >
        {children}
      </TooltipPrimitive.Content>
    </TooltipPrimitive.Portal>
  )
}

interface TipProps extends Omit<React.ComponentProps<typeof TooltipPrimitive.Content>, 'content'> {
  label: React.ReactNode
  children: React.ReactNode
  delayDuration?: number
}

// Drop-in replacement for native `title=`: wrap any single element. Instant,
// position-aware, themed. Self-contained (carries its own Provider) so it works
// anywhere without a provider ancestor. Renders the child untouched when label
// is falsy.
function Tip({ label, children, delayDuration = 0, ...props }: TipProps) {
  // Touch devices have no hover — tooltips fire on tap, fight with primary
  // actions, and add latency. Drop them entirely on the mobile shell.
  if (!label || (typeof window !== 'undefined' && Boolean((window as any).__HERMES_MOBILE_STANDALONE__))) {
    return <>{children}</>
  }

  return (
    <TooltipProvider delayDuration={delayDuration}>
      <Tooltip>
        <TooltipTrigger asChild>{children}</TooltipTrigger>
        <TooltipContent {...props}>{label}</TooltipContent>
      </Tooltip>
    </TooltipProvider>
  )
}

export { Tip, Tooltip, TooltipContent, TooltipProvider, TooltipTrigger }
