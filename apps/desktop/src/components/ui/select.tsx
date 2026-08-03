import { Select as SelectPrimitive } from 'radix-ui'
import * as React from 'react'

import { Codicon } from '@/components/ui/codicon'
import { type ControlVariantProps, controlVariants } from '@/components/ui/control'
import { cn } from '@/lib/utils'

function Select({ ...props }: React.ComponentProps<typeof SelectPrimitive.Root>) {
  return <SelectPrimitive.Root data-slot="select" {...props} />
}

function SelectTrigger({
  className,
  children,
  size,
  ...props
}: React.ComponentProps<typeof SelectPrimitive.Trigger> & ControlVariantProps) {
  return (
    <SelectPrimitive.Trigger
      className={cn(
        controlVariants({ size }),
        'flex items-center justify-between gap-2 whitespace-nowrap data-placeholder:text-muted-foreground [&_svg]:pointer-events-none [&_svg]:shrink-0',
        className
      )}
      data-slot="select-trigger"
      {...props}
    >
      {children}
      <SelectPrimitive.Icon asChild>
        <Codicon className="opacity-60" name="chevron-down" size="1rem" />
      </SelectPrimitive.Icon>
    </SelectPrimitive.Trigger>
  )
}

function SelectValue({ ...props }: React.ComponentProps<typeof SelectPrimitive.Value>) {
  return <SelectPrimitive.Value data-slot="select-value" {...props} />
}

function SelectContent({
  className,
  children,
  position = 'popper',
  ...props
}: React.ComponentProps<typeof SelectPrimitive.Content>) {
  return (
    <SelectPrimitive.Portal>
      <SelectPrimitive.Content
        className={cn(
          'relative z-[140] max-h-72 min-w-32 overflow-hidden rounded-md border border-border bg-popover text-popover-foreground shadow-md data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0 data-[state=closed]:zoom-out-95 data-[state=open]:zoom-in-95 data-[side=bottom]:slide-in-from-top-2 data-[side=left]:slide-in-from-right-2 data-[side=top]:slide-in-from-bottom-2 data-[side=right]:slide-in-from-left-2',
          position === 'popper' &&
            'data-[side=bottom]:translate-y-1 data-[side=left]:-translate-x-1 data-[side=right]:translate-x-1 data-[side=top]:-translate-y-1',
          className
        )}
        data-slot="select-content"
        position={position}
        {...props}
      >
        <SelectPrimitive.Viewport
          className={cn(
            'p-1',
            position === 'popper' && 'h-(--radix-select-trigger-height) w-full min-w-(--radix-select-trigger-width)'
          )}
        >
          {children}
        </SelectPrimitive.Viewport>
      </SelectPrimitive.Content>
    </SelectPrimitive.Portal>
  )
}

function SelectGroup({ ...props }: React.ComponentProps<typeof SelectPrimitive.Group>) {
  return <SelectPrimitive.Group data-slot="select-group" {...props} />
}

function SelectLabel({ className, ...props }: React.ComponentProps<typeof SelectPrimitive.Label>) {
  return (
    <SelectPrimitive.Label
      className={cn('px-2 py-1.5 text-[0.65rem] font-medium uppercase tracking-wide text-muted-foreground', className)}
      data-slot="select-label"
      {...props}
    />
  )
}

function SelectItem({ className, children, ...props }: React.ComponentProps<typeof SelectPrimitive.Item>) {
  return (
    <SelectPrimitive.Item
      className={cn(
        'relative flex w-full cursor-pointer items-center gap-2 rounded-sm py-1.5 pr-8 pl-2 text-xs outline-none select-none focus:bg-accent focus:text-accent-foreground data-disabled:pointer-events-none data-disabled:cursor-default data-disabled:opacity-50',
        className
      )}
      data-slot="select-item"
      {...props}
    >
      <span className="absolute right-2 flex size-3.5 items-center justify-center">
        <SelectPrimitive.ItemIndicator>
          <Codicon name="check" size="1rem" />
        </SelectPrimitive.ItemIndicator>
      </span>
      <SelectPrimitive.ItemText>{children}</SelectPrimitive.ItemText>
    </SelectPrimitive.Item>
  )
}

export { Select, SelectContent, SelectGroup, SelectItem, SelectLabel, SelectTrigger, SelectValue }
