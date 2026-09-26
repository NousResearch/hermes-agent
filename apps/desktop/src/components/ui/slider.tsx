import type { ComponentProps } from 'react'

import { cn } from '@/lib/utils'

// The one range control: a hairline track in the app's stroke tint with the
// native thumb in the primary accent. Settings rows compose it beside a
// readout or inside a labelled group; nothing hand-rolls `type="range"`.
export function Slider({ className, style, ...props }: Omit<ComponentProps<'input'>, 'type'>) {
  return (
    <input
      className={cn('h-1 w-40 cursor-pointer appearance-none rounded-full bg-(--ui-stroke-tertiary)', className)}
      data-slot="slider"
      style={{ accentColor: 'var(--dt-primary)', ...style }}
      type="range"
      {...props}
    />
  )
}
