import { Tabs as TabsPrimitive } from 'radix-ui'
import * as React from 'react'

import { cn } from '@/lib/utils'

function Tabs({ className, ...props }: React.ComponentProps<typeof TabsPrimitive.Root>) {
  return <TabsPrimitive.Root className={cn('flex flex-col gap-2', className)} data-slot="tabs" {...props} />
}

type TabsVisualVariant = 'line' | 'segmented'

type TabsListProps = React.ComponentProps<typeof TabsPrimitive.List> & { variant?: TabsVisualVariant }
type TabsTriggerProps = React.ComponentProps<typeof TabsPrimitive.Trigger> & { variant?: TabsVisualVariant }

function TabsList({ className, variant = 'segmented', ...props }: TabsListProps) {
  return (
    <TabsPrimitive.List
      className={cn(
        variant === 'line'
          ? 'inline-flex h-10 w-full items-stretch justify-start border-b border-(--ui-stroke-tertiary) bg-transparent text-(--ui-text-tertiary)'
          : 'inline-flex h-9 items-center justify-center rounded-lg bg-muted p-1 text-muted-foreground',
        className
      )}
      data-slot="tabs-list"
      {...props}
    />
  )
}

function TabsTrigger({ className, variant = 'segmented', ...props }: TabsTriggerProps) {
  return (
    <TabsPrimitive.Trigger
      className={cn(
        variant === 'line'
          ? 'relative inline-flex h-10 flex-1 items-center justify-center gap-1.5 px-3 text-xs font-medium whitespace-nowrap transition-colors outline-none after:absolute after:inset-x-3 after:-bottom-px after:h-0.5 after:rounded-t after:bg-transparent focus-visible:bg-background focus-visible:text-foreground focus-visible:ring-[0.1875rem] focus-visible:ring-ring/35 disabled:pointer-events-none disabled:opacity-50 data-[state=active]:text-foreground data-[state=active]:after:bg-(--ui-accent)'
          : 'inline-flex h-7 items-center justify-center gap-1.5 rounded-md px-3 text-sm font-medium whitespace-nowrap transition-all outline-none focus-visible:bg-background focus-visible:text-foreground focus-visible:ring-[0.1875rem] focus-visible:ring-ring/35 disabled:pointer-events-none disabled:opacity-50 data-[state=active]:bg-background data-[state=active]:text-foreground data-[state=active]:shadow-xs [&_svg]:pointer-events-none [&_svg]:size-4 [&_svg]:shrink-0',
        className
      )}
      data-slot="tabs-trigger"
      {...props}
    />
  )
}

function TabsContent({ className, ...props }: React.ComponentProps<typeof TabsPrimitive.Content>) {
  return <TabsPrimitive.Content className={cn('min-h-0 flex-1', className)} data-slot="tabs-content" {...props} />
}

export { Tabs, TabsContent, TabsList, TabsTrigger }
