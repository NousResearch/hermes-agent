import { Popover as PopoverPrimitive } from 'radix-ui'
import * as React from 'react'

import { usePopoverPortalContainer } from '@/components/ui/dialog-portal-context'
import { cn } from '@/lib/utils'

function Popover({ ...props }: React.ComponentProps<typeof PopoverPrimitive.Root>) {
  return <PopoverPrimitive.Root data-slot="popover" {...props} />
}

function PopoverTrigger({ ...props }: React.ComponentProps<typeof PopoverPrimitive.Trigger>) {
  return <PopoverPrimitive.Trigger data-slot="popover-trigger" {...props} />
}

function PopoverAnchor({ ...props }: React.ComponentProps<typeof PopoverPrimitive.Anchor>) {
  return <PopoverPrimitive.Anchor data-slot="popover-anchor" {...props} />
}

function PopoverContent({
  align = 'center',
  // Keeps the arrow clear of the rounded corners (rounded-lg = 8px): Radix
  // clamps the arrow this far from each edge and shifts the popover to
  // compensate, so the arrow never jams into a corner on start/end alignment.
  arrowPadding = 12,
  children,
  className,
  collisionPadding = 8,
  sideOffset = 6,
  ...props
}: React.ComponentProps<typeof PopoverPrimitive.Content>) {
  // Portal into the enclosing dialog when nested in one (keeps focus inside so
  // the dialog doesn't close on dismiss); document.body otherwise.
  const container = usePopoverPortalContainer()

  return (
    <PopoverPrimitive.Portal container={container}>
      <PopoverPrimitive.Content
        align={align}
        arrowPadding={arrowPadding}
        // Themed glass surface, viewport-aware (Radix flips/shifts off edges),
        // standard open/close motion. Border-only (no shadow).
        className={cn(
          'z-50 w-72 origin-(--radix-popover-content-transform-origin) rounded-lg border border-(--ui-stroke-secondary) bg-[var(--popover-surface)] p-2 text-popover-foreground backdrop-blur-md outline-hidden data-[side=bottom]:slide-in-from-top-1 data-[side=left]:slide-in-from-right-1 data-[side=right]:slide-in-from-left-1 data-[side=top]:slide-in-from-bottom-1 data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=closed]:zoom-out-95 data-[state=open]:animate-in data-[state=open]:fade-in-0 data-[state=open]:zoom-in-95 [--popover-surface:color-mix(in_srgb,var(--ui-bg-elevated)_92%,transparent)]',
          className
        )}
        collisionPadding={collisionPadding}
        data-slot="popover-content"
        sideOffset={sideOffset}
        {...props}
      >
        {children}
        {/* CSS arrow that truly inherits the surface: a rotated square sharing the
            body's exact bg + backdrop-blur (so it matches even through glass), with
            the border on its two outer edges only. Radix authors the child pointing
            "down" and rotates the wrapper per side, so the V always faces outward.
            The square's inner half tucks under the body, opening the border seam. */}
        <PopoverPrimitive.Arrow asChild height={7} width={16}>
          <span className="relative block h-[7px] w-4 overflow-visible">
            <span className="absolute top-0 left-1/2 size-[11px] -translate-x-1/2 -translate-y-1/2 rotate-45 border-r border-b border-(--ui-stroke-secondary) bg-[var(--popover-surface)] backdrop-blur-md" />
          </span>
        </PopoverPrimitive.Arrow>
      </PopoverPrimitive.Content>
    </PopoverPrimitive.Portal>
  )
}

export { Popover, PopoverAnchor, PopoverContent, PopoverTrigger }
