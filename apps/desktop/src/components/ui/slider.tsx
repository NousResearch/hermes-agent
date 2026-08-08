import { Slider as SliderPrimitive } from 'radix-ui'
import * as React from 'react'

import { cn } from '@/lib/utils'

function Slider({
  className,
  ...props
}: React.ComponentProps<typeof SliderPrimitive.Root>) {
  return (
    <SliderPrimitive.Root
      className={cn(
        'relative flex w-full touch-none select-none items-center data-[orientation=vertical]:h-full data-[orientation=vertical]:w-3 data-[orientation=vertical]:flex-col',
        className
      )}
      data-slot="slider"
      {...props}
    >
      <SliderPrimitive.Track className="relative h-1.5 w-full grow overflow-hidden rounded-full bg-[color-mix(in_srgb,var(--dt-foreground)_14%,transparent)] data-[orientation=vertical]:h-full data-[orientation=vertical]:w-1.5">
        <SliderPrimitive.Range className="absolute h-full bg-primary data-[orientation=vertical]:w-full" />
      </SliderPrimitive.Track>
      <SliderPrimitive.Thumb className="block size-4 rounded-full border border-[color-mix(in_srgb,var(--dt-foreground)_20%,transparent)] bg-foreground shadow-[0_0.0625rem_0.1875rem_color-mix(in_srgb,var(--dt-background)_50%,transparent)] transition-colors focus-visible:outline-none focus-visible:ring-[0.1875rem] focus-visible:ring-ring/50 disabled:pointer-events-none disabled:opacity-50" />
    </SliderPrimitive.Root>
  )
}

export { Slider }
