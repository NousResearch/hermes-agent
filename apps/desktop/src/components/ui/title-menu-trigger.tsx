import type * as React from 'react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { cn } from '@/lib/utils'

/**
 * Compact "Label ▾" chrome trigger. Domain-agnostic — drop in as the child of
 * `DropdownMenuTrigger asChild` (or any asChild menu trigger). Sessions,
 * projects, filters, etc. own their menus; this only owns the trigger look.
 */
export function TitleMenuTrigger({
  children,
  className,
  ...props
}: Omit<React.ComponentProps<typeof Button>, 'children' | 'size' | 'variant'> & {
  children: React.ReactNode
}) {
  return (
    <Button
      className={cn(
        'pointer-events-auto flex h-6 min-w-0 max-w-full gap-1 overflow-hidden border border-transparent bg-transparent px-2 py-0 text-(--ui-text-secondary) hover:border-(--ui-stroke-tertiary) hover:bg-(--ui-control-hover-background) hover:text-foreground data-[state=open]:border-(--ui-stroke-tertiary) data-[state=open]:bg-(--ui-control-active-background) [-webkit-app-region:no-drag]',
        className
      )}
      type="button"
      variant="ghost"
      {...props}
    >
      <span className="min-w-0 flex-1 truncate text-[0.75rem] font-medium leading-none">{children}</span>
      <Codicon className="shrink-0 text-(--ui-text-tertiary)" name="chevron-down" size="0.8125rem" />
    </Button>
  )
}
