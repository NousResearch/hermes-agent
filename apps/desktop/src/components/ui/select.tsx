import { Select as SelectPrimitive } from 'radix-ui'
import * as React from 'react'

import { Codicon } from '@/components/ui/codicon'
import { type ControlVariantProps, controlVariants } from '@/components/ui/control'
import { usePopoverPortalContainer } from '@/components/ui/dialog-portal-context'
import {
  menuItemClass,
  menuItemFocusClass,
  menuLabelClass,
  menuMotionClass,
  menuSurfaceClass
} from '@/components/ui/menu'
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
  collisionPadding = 8,
  position = 'popper',
  sideOffset = 4,
  ...props
}: React.ComponentProps<typeof SelectPrimitive.Content>) {
  // Portal into the enclosing dialog (if any) so the dropdown is a DOM
  // descendant of the dialog — keeps focus inside and stops the dialog closing
  // when the dropdown is dismissed. Falls back to document.body outside a dialog.
  const container = usePopoverPortalContainer()

  return (
    <SelectPrimitive.Portal container={container}>
      <SelectPrimitive.Content
        className={cn(
          menuSurfaceClass,
          menuMotionClass,
          'relative z-(--z-modal-popover) max-h-72 min-w-36 overflow-hidden p-0',
          position === 'popper' &&
            'max-h-[min(18rem,var(--radix-select-content-available-height))] origin-(--radix-select-content-transform-origin)',
          className
        )}
        collisionPadding={position === 'popper' ? collisionPadding : undefined}
        data-slot="select-content"
        position={position}
        sideOffset={position === 'popper' ? sideOffset : undefined}
        {...props}
      >
        <SelectPrimitive.Viewport
          className={cn(
            'dt-portal-scrollbar p-1',
            position === 'popper' && 'w-full min-w-(--radix-select-trigger-width)'
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
  return <SelectPrimitive.Label className={cn(menuLabelClass, className)} data-slot="select-label" {...props} />
}

function SelectItem({ className, children, ...props }: React.ComponentProps<typeof SelectPrimitive.Item>) {
  return (
    <SelectPrimitive.Item
      className={cn(
        menuItemClass,
        menuItemFocusClass,
        'w-full cursor-pointer pr-7 data-disabled:pointer-events-none data-disabled:cursor-default data-disabled:opacity-50',
        className
      )}
      data-slot="select-item"
      {...props}
    >
      <span className="absolute right-2 flex size-3.5 items-center justify-center text-foreground">
        <SelectPrimitive.ItemIndicator>
          <Codicon name="check" size="0.75rem" />
        </SelectPrimitive.ItemIndicator>
      </span>
      <SelectPrimitive.ItemText>{children}</SelectPrimitive.ItemText>
    </SelectPrimitive.Item>
  )
}

export { Select, SelectContent, SelectGroup, SelectItem, SelectLabel, SelectTrigger, SelectValue }
