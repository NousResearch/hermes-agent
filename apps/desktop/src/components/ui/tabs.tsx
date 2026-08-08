import { Tabs as TabsPrimitive } from 'radix-ui'
import * as React from 'react'

import { cn } from '@/lib/utils'

function Tabs({ className, ...props }: React.ComponentProps<typeof TabsPrimitive.Root>) {
  return <TabsPrimitive.Root className={cn('flex flex-col gap-2', className)} data-slot="tabs" {...props} />
}

function TabsList({ className, ...props }: React.ComponentProps<typeof TabsPrimitive.List>) {
  return (
    <TabsPrimitive.List
      className={cn(
        'inline-flex h-9 items-center justify-center rounded-[var(--radius-sm)] border border-(--ui-stroke-tertiary) bg-[color-mix(in_srgb,var(--ui-bg-elevated)_86%,transparent)] p-1 text-muted-foreground shadow-xs',
        className
      )}
      data-slot="tabs-list"
      {...props}
    />
  )
}

function TabsTrigger({ className, ...props }: React.ComponentProps<typeof TabsPrimitive.Trigger>) {
  return (
    <TabsPrimitive.Trigger
      className={cn(
        'inline-flex h-7 items-center justify-center gap-1.5 rounded-[var(--radius-sm)] px-3 text-[0.8125rem] font-medium whitespace-nowrap transition-[background-color,box-shadow,color,transform] duration-200 ease-[cubic-bezier(0.2,0.8,0.2,1)] outline-none hover:bg-(--chrome-action-hover) hover:text-foreground active:scale-[0.98] focus-visible:ring-[0.1875rem] focus-visible:ring-ring/35 disabled:pointer-events-none disabled:opacity-50 data-[state=active]:bg-background data-[state=active]:text-foreground data-[state=active]:shadow-sm [&_svg]:pointer-events-none [&_svg]:size-4 [&_svg]:shrink-0',
        className
      )}
      data-slot="tabs-trigger"
      {...props}
    />
  )
}

export { Tabs, TabsList, TabsTrigger }
