import { Tooltip as TooltipPrimitive } from 'radix-ui'
import * as React from 'react'

import { cn } from '@/lib/utils'

function TooltipProvider({
  delayDuration = 0,
  // Tips are labels, not interactive surfaces. Hoverable content + Radix's
  // pointer-grace bridge is what leaves tips stuck open — especially over
  // Electron `-webkit-app-region: drag` chrome where pointermove never fires
  // to clear the grace area. Default off so open state tracks the trigger only.
  disableHoverableContent = true,
  ...props
}: React.ComponentProps<typeof TooltipPrimitive.Provider>) {
  return (
    <TooltipPrimitive.Provider
      data-slot="tooltip-provider"
      delayDuration={delayDuration}
      disableHoverableContent={disableHoverableContent}
      {...props}
    />
  )
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
        // Transparent, width-capped wrapper. The visible chip is the inner inline
        // span so `box-decoration-break: clone` gives a marker-style background
        // that hugs EACH wrapped line (bg only on the text, ragged right — no
        // rectangular dead space). Instant, no transition (delayDuration=0).
        // pointer-events-none: the tip must never steal hover/clicks from the
        // chrome underneath (titlebar tools, adjacent tabs, etc.).
        className={cn('pointer-events-none z-[200] w-fit max-w-64 select-none', className)}
        data-slot="tooltip-content"
        sideOffset={sideOffset}
        {...props}
      >
        {/* bg-foreground/text-background auto-inverts per theme. leading-normal
            keeps lines readable; py-1 makes the cloned line-boxes overlap just
            enough to read as one continuous fill (no gaps between lines). */}
        <span className="box-decoration-clone inline bg-foreground px-1.5 py-1 text-[11px] font-bold leading-normal text-background [font-family:Arial,sans-serif]">
          {children}
        </span>
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
// is falsy. Open state is trigger-hover only — never sticky, never click-blocking.
function Tip({ label, children, delayDuration = 0, ...props }: TipProps) {
  if (!label) {
    return <>{children}</>
  }

  return (
    <TooltipProvider delayDuration={delayDuration} disableHoverableContent>
      <Tooltip disableHoverableContent>
        <TooltipTrigger asChild>{children}</TooltipTrigger>
        <TooltipContent {...props}>{label}</TooltipContent>
      </Tooltip>
    </TooltipProvider>
  )
}

export { Tip, Tooltip, TooltipContent, TooltipProvider, TooltipTrigger }
